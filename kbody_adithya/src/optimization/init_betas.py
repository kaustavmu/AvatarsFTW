from pytorch_lightning.callbacks import Callback

import numpy as np
import logging

log = logging.getLogger(__name__)

__all__ = ["InitBetasExpose"]

class InitBetasExpose(Callback):
    def __init__(self):
        super().__init__()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx: int, unused=0) -> None:
        res = batch['betas']
        for e, p in zip(res[0], pl_module.preprocess.betas.parameters()):
            p.data = e[np.newaxis, np.newaxis].clone()
        log.info("Initialized beta parameters from ExPose inference results.")
