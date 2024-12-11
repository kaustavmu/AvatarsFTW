from collections.abc import Callable

from scipy.ndimage import distance_transform_edt

import math
import cv2
import numpy as np
import typing
import torch
import logging
import os

log = logging.getLogger(__name__)

__all__ = [
    'SilhouetteHoles',
]

class SilhouetteHoles(Callable):
    def __init__(self,
        mask_threshold:     float=0.5,
        kernel_size:        int=5,
        iterations:         int=3,
        symmetric:          bool=True,
        inverse:            bool=False,
        scale_inner:        float=0.5,
        scale_outer:        float=2.0,
        max_area_threshold: float=2.5e-3,
        hull_distance_test: float=1e-3,
    ) -> None:
        super().__init__()
        self.mask_threshold = mask_threshold
        self.ksize = (kernel_size, kernel_size)
        self.iters = iterations
        self.symmetric = symmetric
        self.inverse = inverse
        self.scale_inner = scale_inner
        self.scale_outer = scale_outer
        self.max_area_threshold = max_area_threshold
        self.hull_distance_test = hull_distance_test
        log.info(f"Using a distance field scaled by [{scale_inner}] (inner) and [{scale_outer}] (outer).")

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

    def calc_hole_mask(self, img: torch.Tensor, **params) -> np.ndarray:
        b = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        _, b = cv2.threshold(b, int(self.mask_threshold * 255), 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(b, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE )
        # image = np.zeros((b.shape[0], b.shape[1], 3))
        mask = np.zeros((b.shape[0], b.shape[1], 1))
        if hierarchy is None:
            return mask # image, mask
        hierarchy = hierarchy[0]
        areas = np.array([cv2.contourArea(c) for c in contours])
        max_area = areas.max()
        holes = []
        for i, c in enumerate(contours):
            if areas[i] == max_area:
                # cv2.drawContours(image, contours, i, (0, 0, 255), 2)
                hull = cv2.convexHull(c)
                silhouette = c
            else: # areas[i] > (mean_area - std_area):
                # cv2.drawContours(image, contours, i, (0, 255, 0), 1)
                holes.append((c, areas[i]))
        diag = math.sqrt(b.shape[0] ** 2 + b.shape[1] ** 2)
        num_holes = 0
        for h, a in holes:            
            if a > self.max_area_threshold * max_area and all((                
                cv2.pointPolygonTest(
                    silhouette, p.squeeze().astype(np.float32), True
                ) > self.hull_distance_test * diag for p in h
            )):
                # cv2.drawContours(image, [h], 0, (255, 0, 120), 3)
                cv2.fillPoly(mask, [h], 255)
                num_holes += 1
        log.info(f"Found {num_holes} closed internal contours that will be penalized.")
        return mask # return image, mask

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
        holes = self.calc_hole_mask(matte) / 255.0
        holes = torch.from_numpy(holes.transpose(2, 0, 1))
        mask = (matte > self.mask_threshold).float()
        # edt = self.edt(self.gradient(self.erosion(mask.numpy().squeeze())))
        edt = self.edt(mask.numpy().squeeze())
        return {
            'matte': matte.to(device)[np.newaxis, ...].float(),
            'silhouette': mask.to(device)[np.newaxis, ...],
            'holes': holes.to(device)[np.newaxis, ...],
            'edt': torch.from_numpy(edt[np.newaxis, ...]).float().to(device)[np.newaxis, ...],
        }