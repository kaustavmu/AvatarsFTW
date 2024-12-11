from ts.utils.util import PredictionException

import torch
import functools
import typing
import os

__all__ = ['IoU']

def _dim_list(
    tensor:         torch.Tensor,
    start_index:    int=1,
) -> typing.List[int]:
    return list(range(start_index, len(tensor.shape)))

spatial_dim_list = functools.partial(_dim_list, start_index=2)

class IoU(torch.nn.Module):
    def __init__(self, 
        threshold:  float=0.5,
        # throw:      bool=True,
    ):
        super(IoU, self).__init__()
        self.reduce = True
        self.threshold = threshold
        self.throw = bool(int(os.environ.get('KLOTHED_FIT_THROW_ON_ERROR', True)))

    def forward(self, 
        pred:   torch.Tensor,
        gt:     torch.Tensor,        
    ) -> torch.Tensor:
        if pred is None and gt is None:
            return torch.scalar_tensor(0.0)
        dims = spatial_dim_list(pred)
        intersect = (pred * gt).sum(dims)
        union = (pred + gt - (pred * gt)).sum(dims) + 1e-6
        iou = (intersect / union).sum() / intersect.numel() #NOTE: is correct for batch size = 1 only
        if iou < self.threshold and self.throw:
            raise PredictionException(f"Bad quality fit (iou={float(iou)})", 512)
        return iou