import cached_conv as cc
from hydra import compose, initialize
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

try :
    cfg = OmegaConf.load('../params.yaml')
except:
    cfg = OmegaConf.load('params.yaml')

# Instantiate cached convolution modules as partial functions
cc.Conv1d = instantiate(cfg.model.cc.Conv1d)
cc.ConvTranspose1d = instantiate(cfg.model.cc.ConvTranspose1d)
cc.get_padding = instantiate(cfg.model.cc.get_padding)

from .blocks import *
from .discriminator import *
from .model import RAVE, BetaWarmupCallback
from .pqmf import *