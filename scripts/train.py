import hashlib
import os
import sys
from typing import Any, Dict
import copy

# import gin
import hydra
from hydra.utils import instantiate
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

try:
    import rave
except:
    import sys, os 
    sys.path.append(os.path.abspath('.'))
    import rave

import rave
import rave.core
import rave.dataset
from rave.transforms import get_augmentations, add_augmentation

class EMA(pl.Callback):

    def __init__(self, factor=.999) -> None:
        super().__init__()
        self.weights = {}
        self.factor = factor

    def on_train_batch_end(self, trainer, pl_module, outputs, batch,
                           batch_idx) -> None:
        for n, p in pl_module.named_parameters():
            if n not in self.weights:
                self.weights[n] = p.data.clone()
                continue

            self.weights[n] = self.weights[n] * self.factor + p.data * (
                1 - self.factor)

    def swap_weights(self, module):
        for n, p in module.named_parameters():
            current = p.data.clone()
            p.data.copy_(self.weights[n])
            self.weights[n] = current

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        if self.weights:
            self.swap_weights(pl_module)
        else:
            print("no ema weights available")

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self.weights:
            self.swap_weights(pl_module)
        else:
            print("no ema weights available")

    def state_dict(self) -> Dict[str, Any]:
        return self.weights.copy()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.weights.update(state_dict)

@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(cfg):
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    # create model
    model = instantiate(cfg.model.rave.RAVE, _recursive_=True)

    if cfg.train.derivative:
        model.integrator = rave.dataset.get_derivator_integrator(model.sr)[1]

    # parse datasset
    dataset = rave.dataset.get_dataset(cfg.train.db_path,
                                       model.sr,
                                       cfg.train.n_signal,
                                       derivative=cfg.train.derivative,
                                       normalize=cfg.train.normalize,
                                       rand_pitch=cfg.train.rand_pitch,
                                       n_channels=cfg.train.channels)
    
    train, val = rave.dataset.split_dataset(dataset, 98, max_residual=cfg.model.dataset.split_dataset.max_residual)

    # get data-loader
    num_workers = cfg.train.workers
    if os.name == "nt" or sys.platform == "darwin":
        num_workers = 0
    train = DataLoader(train,
                       cfg.train.batch,
                       True,
                       drop_last=True,
                       num_workers=num_workers)
    val = DataLoader(val, cfg.train.batch, False, num_workers=num_workers)

    # CHECKPOINT CALLBACKS
    validation_checkpoint = pl.callbacks.ModelCheckpoint(monitor="validation",
                                                         filename="best")
    last_filename = "last" if cfg.train.save_every is None else "epoch-{epoch:04d}"                                                        
    last_checkpoint = rave.core.ModelCheckpoint(filename=last_filename, step_period=cfg.train.save_every)

    val_check = {}
    if len(train) >= cfg.train.val_every:
        val_check["val_check_interval"] = 1 if cfg.train.smoke_test else cfg.train.val_every
    else:
        nepoch = cfg.train.val_every // len(train)
        val_check["check_val_every_n_epoch"] = nepoch

    if cfg.train.smoke_test:
        val_check['limit_train_batches'] = 1
        val_check['limit_val_batches'] = 1

    RUN_NAME = f'{cfg.train.name}'

    os.makedirs(os.path.join(cfg.train.out_path, RUN_NAME), exist_ok=True)

    if cfg.train.gpu == -1:
        gpu = 0
    else:
        gpu = cfg.train.sgpu or rave.core.setup_gpu()

    print('selected gpu:', gpu)

    accelerator = None
    devices = None
    if cfg.train.gpu == -1:
        pass
    elif torch.cuda.is_available():
        accelerator = "cuda"
        devices = cfg.train.gpu or rave.core.setup_gpu()
    elif torch.backends.mps.is_available():
        print(
            "Training on mac is not available yet. Use --gpu -1 to train on CPU (not recommended)."
        )
        exit()
        accelerator = "mps"
        devices = 1

    callbacks = [
        validation_checkpoint,
        last_checkpoint,
        rave.model.WarmupCallback(),
        rave.model.QuantizeCallback(),
        # rave.core.LoggerCallback(rave.core.ProgressLogger(RUN_NAME)),
        instantiate(cfg.model.rave.BetaWarmupCallback, _recursive_=True),
    ]

    if cfg.train.ema is not None:
        callbacks.append(EMA(cfg.train.ema))

    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(
            cfg.train.out_path,
            name=RUN_NAME,
        ),
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        max_epochs=300000,
        max_steps=cfg.train.max_steps,
        profiler="simple",
        enable_progress_bar=cfg.train.progress,
        **val_check,
    )

    run = rave.core.search_for_run(cfg.train.ckpt)
    if run is not None:
        print('loading state from file %s'%run)
        loaded = torch.load(run, map_location='cpu')
        # model = model.load_state_dict(loaded)
        trainer.fit_loop.epoch_loop._batches_that_stepped = loaded['global_step']
        # model = model.load_state_dict(loaded['state_dict'])
    
    # with open(os.path.join(cfg.train.out_path, RUN_NAME, "config.gin"), "w") as config_out:
    #     config_out.write(gin.operative_config_str())

    trainer.fit(model, train, val, ckpt_path=run)


if __name__ == "__main__": 
    main()
