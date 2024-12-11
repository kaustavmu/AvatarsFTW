try:
    from losses.L2 import L2
except (ModuleNotFoundError, ImportError) as e:
    from L2 import L2

import torch

__all__ = ["GemanMcClure"]

class GemanMcClure(L2):
    r"""Implements the Geman-McClure error function.

    """
    def __init__(self,
        rho: float=100.0,
    ):
        super(GemanMcClure, self).__init__()
        self.rho_sq = rho ** 2

    def forward(self,
        pred:       torch.Tensor,
        gt:         torch.Tensor=None,
        weights:    torch.Tensor=None, # float tensor
        mask:       torch.Tensor=None, # byte tensor
    ) -> torch.Tensor:
        L2 = super(GemanMcClure, self).forward(pred=pred, gt=gt)\
            if gt is not None else pred
        gm = L2 / (L2 + self.rho_sq) * self.rho_sq
        if weights is not None:
            gm = gm * weights
        if mask is not None:
            gm = gm[mask]
        return gm