from collections.abc import Callable

import typing
import torch
import logging
import cv2

__all__ = [
    'ImageFileInput',
]

log = logging.getLogger(__name__)

def load_color_image(
    filename:       str,    
    output_space:   str='norm', # ['norm', 'ndc', 'pixel']
) -> torch.Tensor:
    img = torch.from_numpy(
        cv2.imread(filename).transpose(2, 0, 1)
    ).flip(dims=[0])
    
    if output_space == 'norm':
        img = img / 255.0
    return img

class ImageFileInput(Callable):
    def __init__(self,
        input_key:          str='image',
        output_key:         str='color',
    ) -> None:
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key

    def __call__(self, 
        data:   typing.Mapping[str, typing.Any],
        device: torch.device,
    ) -> torch.Tensor:
        path = data.get(self.input_key)
        log.info(f"Loading image file [key: {self.input_key}] @ {path}")
        return {self.output_key: load_color_image(path).unsqueeze(0).to(device) }
