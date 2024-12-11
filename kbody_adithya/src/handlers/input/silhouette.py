from collections.abc import Callable

from scipy.ndimage import distance_transform_edt

import cv2
import numpy as np
import typing
import torch
import logging
import os

log = logging.getLogger(__name__)

__all__ = ['Silhouette']

class Silhouette(Callable):
    def __init__(self,
        mask_threshold:     float=0.5,
        kernel_size:        int=5,
        iterations:         int=3,
        symmetric:          bool=True,
        inverse:            bool=False,
        scale_inner:        float=0.5,
        scale_outer:        float=2.0,
    ) -> None:
        super().__init__()
        self.mask_threshold = mask_threshold
        self.ksize = (kernel_size, kernel_size)
        self.iters = iterations
        self.symmetric = symmetric
        self.inverse = inverse
        self.scale_inner = scale_inner
        self.scale_outer = scale_outer

    def erosion(self, img: np.ndarray) -> np.ndarray:
        element = cv2.getStructuringElement(cv2.MORPH_RECT, self.ksize, (-1, -1))
        eroded = cv2.erode(img, element, iterations=self.iters, borderType=cv2.BORDER_CONSTANT)
        return eroded[..., np.newaxis] if len(eroded.shape) == 2 else eroded

    def gradient(self, img: np.ndarray) -> np.ndarray:        
        element = cv2.getStructuringElement(cv2.MORPH_RECT, self.ksize, (-1, -1))
        grad = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, element, iterations=self.iters, borderType=cv2.BORDER_CONSTANT)
        return grad[..., np.newaxis] if len(grad.shape) == 2  else grad

    def inverse_edt(self, img: np.ndarray) -> np.ndarray:
        binary = np.logical_not(img)
        return binary * distance_transform_edt(binary)

    def edt(self, img: np.ndarray, **params) -> np.ndarray:
        b = img.astype(np.bool8)
        if self.inverse:
            b = np.logical_not(b)
        edt = distance_transform_edt(b)
        return edt if not self.symmetric else b * edt * self.scale_inner + self.inverse_edt(b) * self.scale_outer
        # b = np.logical_not(b)
        # edt = distance_transform_edt(b)
        # return edt

    def __call__(self, 
        data:   typing.Mapping[str, typing.Any],
        device: torch.device,
    ) -> typing.Dict[str, torch.Tensor]:
        filename = data.get('matte')
        log.info(f"Loading [matte] file @ {filename}")
        matte = torch.from_numpy(
            cv2.imread(filename).transpose(2, 0, 1)
        )[0:1, ...]
        if not os.path.splitext(filename)[-1] == '.exr':
            matte = matte / 255.0
        mask = (matte > self.mask_threshold).float()
        # edt = self.edt(self.gradient(self.erosion(mask.numpy().squeeze())))
        edt = self.edt(mask.numpy().squeeze())
        return {
            'matte': matte.to(device)[np.newaxis, ...],
            'silhouette': mask.to(device)[np.newaxis, ...],
            'edt': torch.from_numpy(edt[np.newaxis, ...]).float().to(device)[np.newaxis, ...],
        }