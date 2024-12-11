import os
import torch
from ts.utils.util import PredictionException

__all__ = ['FitValidator']

class FitValidator(torch.nn.Module):
    def __init__(self, 
        # throw:  bool=True,
    ):
        super().__init__()
        self.throw = bool(int(os.environ.get('KLOTHED_FIT_THROW_ON_ERROR', True)))

    def forward(self, 
        translation:        torch.Tensor,
        rotation:           torch.Tensor,
        betas:              torch.Tensor,
        # iou:                torch.Tensor,
    ) -> torch.Tensor:
        # if iou.item() < 0.5:
        #     return torch.scalar_tensor(-1.0).float()
        if torch.isnan(betas).any():
            if self.throw:
                raise PredictionException(f"Numerical optimization error", 513)
            else:
                return torch.scalar_tensor(-2.0).float()
        elif translation[:, 2] < 0.3:            
            if self.throw:
                raise PredictionException("Negative translation error", 514)
            else:
                return torch.scalar_tensor(-3.0).float()
        else:
            return torch.scalar_tensor(1.0).float()