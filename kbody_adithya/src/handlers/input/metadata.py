from collections.abc import Callable

import numpy as np
import typing
import torch
import logging
import os
import cv2

log = logging.getLogger(__name__)

__all__ = ['Metadata']

class Metadata(Callable):
    def __init__(self,
        focal_length:           float=5000.0,
    ) -> None:
        super().__init__()
        self.focal_length = focal_length

    def __call__(self, 
        data:   typing.Mapping[str, typing.Any],
        device: torch.device,
    ) -> typing.Dict[str, torch.Tensor]:
        filename = data.get('metadata_t', '')
        if filename is not None and os.path.exists(filename):
            metadata = np.load(filename, allow_pickle=True)['metadata'].item()
            bottom_pad = metadata.get('bottom_padding', 0)
            log.info(f"Loading metadata from: {filename}, bottom pad = {bottom_pad}.")
        else:
            log.warning(f"No metadata have been found @ {filename}, does not exist, proceeding nominal camera parameters.")
            bottom_pad = 0
        fx, fy = self.focal_length, self.focal_length
        h, w, _ = cv2.imread(data.get('image')).shape
        return { 
            'camera_intrinsics': torch.Tensor(
                [
                    [fx, 0.0, w / 2.0],
                    [0.0, fy, (h - bottom_pad) / 2.0],
                    [0.0, 0.0, 1.0],
                ]
            ).float()[np.newaxis, ...].to(device),
        }
