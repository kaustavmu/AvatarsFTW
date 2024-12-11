import kornia
import torch

class Sobel(kornia.filters.Sobel):
    def __init__(self,        
        normalized:     bool=False,
        epsilon:        int=1e-6,
    ):
        super(Sobel, self).__init__(
            normalized=normalized, eps=epsilon,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return super(Sobel, self).forward(image)