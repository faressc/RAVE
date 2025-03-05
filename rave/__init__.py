# from pathlib import Path

import cached_conv as cc
# import gin
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

# BASE_PATH: Path = Path(__file__).parent

# gin.add_config_file_search_path(BASE_PATH)
# gin.add_config_file_search_path(BASE_PATH.joinpath('configs'))
# gin.add_config_file_search_path(BASE_PATH.joinpath('configs', 'augmentations'))


# def __safe_configurable(name):
#     try: 
#         setattr(cc, name, gin.get_configurable(f"cc.{name}"))
#     except ValueError:
#         setattr(cc, name, gin.external_configurable(getattr(cc, name), module="cc"))

# cc.get_padding = gin.external_configurable(cc.get_padding, module="cc")
# cc.Conv1d = gin.external_configurable(cc.Conv1d, module="cc")
# cc.ConvTranspose1d = gin.external_configurable(cc.ConvTranspose1d, module="cc")

# __safe_configurable("get_padding")
# __safe_configurable("Conv1d")
# __safe_configurable("ConvTranspose1d")

# if GlobalHydra.instance() is not None:
#     GlobalHydra.instance().clear()

# config_path = '../conf'
# initialize(config_path=config_path, version_base="1.1")
# cfg = compose(config_name="config")
try :
    cfg = OmegaConf.load('../params.yaml')
except:
    cfg = OmegaConf.load('params.yaml')

# Instantiate cached convolution modules as partial functions
cc.Conv1d = instantiate(cfg.model.cc.Conv1d)
cc.ConvTranspose1d = instantiate(cfg.model.cc.ConvTranspose1d)
cc.get_padding = instantiate(cfg.model.cc.get_padding)

# if GlobalHydra.instance() is not None:
#     GlobalHydra.instance().clear()

from .blocks import *
from .discriminator import *
from .model import RAVE, BetaWarmupCallback
from .pqmf import *
from .balancer import *