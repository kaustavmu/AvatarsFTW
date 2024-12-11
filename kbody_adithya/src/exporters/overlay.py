from src.handlers.output import Renderer as OverlayRenderer

import torch
import typing

__all__ = ['Renderer']

class Renderer(OverlayRenderer):
    def __init__(self,
    ) -> None:
        super().__init__()
        self.index = 0

    def __call__(self, 
        tensors: typing.Dict[str, torch.Tensor],
        step:       typing.Optional[int]=None,
    ) -> None:
        b = tensors['joints2d'].shape[0]
        super().__call__(tensors, [{
                'body': {
                    'overlay_t': f"overlay_{self.index + i}.jpg",
                }
            } for i in range(b)
        ])
        self.index = self.index + b