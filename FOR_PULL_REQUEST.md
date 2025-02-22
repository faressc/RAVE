# For pull request

## num_signal issue

- in the preprocessing step, the `num_signal` parameter is multiplied by 4 to read one chunk in the load_audio_chunk function. But as one sample is int16, the `num_signal` parameter should be multiplied by 2 instead of 4. This also removes the necessity to multiply the n_seconds parameter by 2 to get the correct dataset length.
