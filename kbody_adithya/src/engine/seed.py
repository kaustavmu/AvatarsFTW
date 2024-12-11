import random
import os
import numpy as np
import torch
import logging

log = logging.getLogger(__name__)

#NOTE: modified from https://github.com/wbaek/torchskeleton/blob/master/skeleton/utils/utils.py
#       and https://github.com/PyTorchLightning/PyTorch-Lightning/blob/0.7.6/pytorch_lightning/trainer/seed.py#L32 

__all__ = ["ManualSeed"]

class ManualSeed(object):
    def __init__(self, 
        seed: int,
        deterministic: bool
    ):
        log.info(f"Setting the seed to {seed}.")
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ["PL_GLOBAL_SEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        #pl.seed_everything(seed)
        if deterministic:
            # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' #NOTE: see https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility 
            # torch.set_deterministic(True) #NOTE: see https://pytorch.org/docs/stable/generated/torch.set_deterministic.html#torch.set_deterministic 
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False