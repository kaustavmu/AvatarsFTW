from src.handlers.output import (
    BodySerializer,
    BodyLegacySerializer,
)

import torch
import typing

__all__ = [
    'BodyPkl',
    'BodyLegacyPkl',
]

class BodyPkl(BodySerializer):
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
                    'body_t': f"body_{self.index + i}.pkl",
                }
            } for i in range(b)
        ])
        self.index = self.index + b

class BodyLegacyPkl(BodyLegacySerializer):
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
                    'body_legacy_t': f"body_legacy_{self.index + i}.pkl",
                }
            } for i in range(b)
        ])
        self.index = self.index + b