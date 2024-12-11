import torch
import logging

__all__ = [
    'DisableCuDNN',
    'DisableTensorCores',
]

log = logging.getLogger(__name__)

class DisableCuDNN(object):
    def __init__(self):
        log.info("Disabling CuDNN.")
        torch.backends.cudnn.enabled = False

class DisableTensorCores(object):
    def __init__(self):
        log.info("Disabling Tensor Cores.")
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False