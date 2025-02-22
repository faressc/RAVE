import functools
import multiprocessing
import os
import pathlib
import subprocess
from datetime import timedelta
from functools import partial
from itertools import repeat
from typing import Callable, Iterable, Sequence, Tuple
import argparse
import math

import lmdb
import numpy as np
import torch
import yaml
from tqdm import tqdm
try:
    from proto.audio_example_pb2 import AudioExample
except:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from proto.audio_example_pb2 import AudioExample

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()

parser.add_argument('--input_path', type=str, help='Path to a directory containing audio files', required=True)
parser.add_argument('--output_path', type=str, help='Output directory for the dataset', required=True)
parser.add_argument('--num_signal', type=int, help='Number of audio samples to use during training', default=131072)
parser.add_argument('--channels', type=int, help="Number of audio channels", required=True)
parser.add_argument('--sample_rate', type=int, help='Sampling rate to use during training', default=44100)
parser.add_argument('--max_db_size', type=int, help='Maximum size (in GB) of the dataset', default=100)
parser.add_argument('--ext', type=str, nargs='+', help='Extension to search for in the input directory', default=['aif', 'aiff', 'wav', 'opus', 'mp3', 'aac', 'flac', 'ogg'])

def float_array_to_int16_bytes(x):
    return np.floor(x * (2**15 - 1)).astype(np.int16).tobytes()


def load_audio_chunk(path: str, n_signal: int,
                     sr: int, channels: int = 1) -> Iterable[np.ndarray]:

    _, wav_channels = get_audio_channels(path)
    channel_map = range(channels)
    if wav_channels < channels:
        channel_map = (math.ceil(channels / wav_channels) * list(range(wav_channels)))[:channels]
    
    processes = []
    for i in range(channels): 
        process = subprocess.Popen(
            [
                'ffmpeg', '-hide_banner', '-loglevel', 'panic', '-i', path, 
                '-ar', str(sr),
                '-f', 's16le',
                '-filter_complex', 'channelmap=%d-0'%channel_map[i],
                '-'
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        processes.append(process)
    
    # Times two because we are reading 2 bytes per sample (16 bit)
    chunk = [p.stdout.read(n_signal * 2) for p in processes]
    while len(chunk[0]) == n_signal * 2:
        yield b''.join(chunk)
        chunk = [p.stdout.read(n_signal * 2) for p in processes]
    process.stdout.close()
    process.stderr.close()


def get_audio_length(path: str) -> float:
    process = subprocess.Popen(
        [
            'ffprobe', '-i', path, '-v', 'error', '-show_entries',
            'format=duration'
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    if process.returncode: return None
    try:
        stdout = stdout.decode().split('\n')[1].split('=')[-1]
        length = float(stdout)
        return path, float(length)
    except:
        return None

def get_audio_channels(path: str) -> int:
    process = subprocess.Popen(
        [
            'ffprobe', '-i', path, '-v', 'error', '-show_entries',
            'stream=channels'
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, _ = process.communicate()
    if process.returncode: return None
    try:
        stdout = stdout.decode().split('\n')[1].split('=')[-1]
        channels = int(stdout)
        return path, int(channels)
    except:
        return None


def flatten(iterator: Iterable):
    for elm in iterator:
        for sub_elm in elm:
            yield sub_elm

def get_metadata(audio_samples, channels: int = 1):
    audio = np.frombuffer(audio_samples, dtype=np.int16)
    audio = audio.astype(float) / (2**15 - 1)
    audio = audio.reshape(channels, -1)
    peak_amplitude = np.amax(np.abs(audio))
    rms_amplitude = np.sqrt(np.mean(audio**2))
    return {'peak': peak_amplitude, 'rms_amplitude': rms_amplitude}


def process_audio_array(audio: Tuple[int, bytes],
                        env: lmdb.Environment,
                        sample_rate: int = 44100,
                        channels: int = 1) -> int:
    audio_id, audio_samples = audio
    buffers = {}
    buffers['waveform'] = AudioExample.AudioBuffer(
        shape=(channels, int(len(audio_samples) / channels)),
        sampling_rate=sample_rate,
        data=audio_samples,
        precision=AudioExample.Precision.INT16,
    )

    ae = AudioExample(buffers=buffers)
    # The :08d means that the string will be 8 characters long and will be zero-padded
    key = f'{audio_id:08d}'
    with env.begin(write=True) as txn:
        txn.put(
            key.encode(),
            ae.SerializeToString(),
        )
    return audio_id

def flatmap(pool,
            func: Callable,
            iterable: Iterable,
            chunksize=None):
    queue = multiprocessing.Manager().Queue(maxsize=os.cpu_count())
    pool.map_async(
        functools.partial(flat_mappper, func),
        zip(iterable, repeat(queue)),
        chunksize,
        lambda _: queue.put(None),
        lambda *e: print(e),
    )

    item = queue.get()
    while item is not None:
        yield item
        item = queue.get()


def flat_mappper(func, arg):
    data, queue = arg
    for item in func(data):
        queue.put(item)


def search_for_audios(path: str, extensions: Sequence[str]):
    path = pathlib.Path(path)
    audios = []
    if not path.is_dir():
        raise ValueError(f"Path {path} is not a directory")
    for ext in extensions:
        audios.append(path.rglob(f'*.{ext}'))
        audios.append(path.rglob(f'*.{ext.upper()}'))
    # flatten because rglob returns a generator and we list them for each extension
    audios = flatten(audios)
    return audios


def main(argv):
    args = parser.parse_args()

    print(f"Processing audio files in {args.input_path} to {args.output_path}")

    # The LMDB database will itself create the final directory, as it has multiple files
    # The * operator is used to unpack the tuple into single elements
    output_dir = os.path.join(*os.path.split(args.output_path)[:-1])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    print(f"Creating LMDB database at {args.output_path}")
    # Create a new LMDB database
    env = lmdb.open(
        args.output_path,
        map_size=args.max_db_size * 1024**3,
    )

    print("Searching for audio files")
    # Search for audio files
    audios = search_for_audios(args.input_path, args.ext)
    audios = map(str, audios)
    audios = map(os.path.abspath, audios)
    # Evaluate the generator
    audios = list(audios)
    print("Number of audio files: ", len(audios))
    if len(audios) == 0:
        print("No valid file found in %s. Aborting"%args.input_path)

    print("Loading audio files")
    # Fix the parameters for the load_audio_chunk function in new chunk_load function
    chunk_load = partial(load_audio_chunk,
                         n_signal=args.num_signal,
                         sr=args.sample_rate,
                         channels=args.channels)
    # create a pool of workers
    pool = multiprocessing.Pool()
    # load chunks
    chunks = flatmap(pool, chunk_load, audios)
    chunks = enumerate(chunks)
    # The map function will not be evaluated until we iterate over it
    processed_samples = map(partial(process_audio_array, env=env, sample_rate = args.sample_rate, channels=args.channels), chunks)
    
    # Evaluate the generator
    pbar = tqdm(processed_samples)
    n_seconds = 0
    for audio_id in pbar:
        n_seconds = (args.num_signal) / args.sample_rate * audio_id
        pbar.set_description(
            f'Current dataset length: {timedelta(seconds=n_seconds)}')
    pbar.close()

    with open(os.path.join(
            args.output_path,
            'metadata.yaml',
    ), 'w') as metadata:
        yaml.safe_dump({'channels': args.channels, 'n_seconds': n_seconds, 'sr': args.sample_rate}, metadata)
    pool.close()
    env.close()

if __name__ == '__main__':
    main(sys.argv)
