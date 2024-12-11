from src.handlers.output import KlothedBodyV2

import os
import glob
import torch
import typing
import logging

log = logging.getLogger(__name__)

__all__ = ['KlothedV2']

class KlothedV2(KlothedBodyV2):
    def __init__(self,
        focal_length:               typing.Union[float, typing.Tuple[float, float]]=5000.0,
        principal_point:            typing.Optional[typing.Union[float, typing.Tuple[float, float]]]=None,
        scale:                      float=1.0,
        blend:                      float=0.65,
        pad_scale:                  float=2.5,
        shoulder_scale:             float=0.75,
        landmark_perc_threshold:    float=0.1,
        joints2d:                   str='joints2d',
        joints3d:                   str='smplx_joints',
        j3d_head_index:             int=0,
        mirror:                     bool=False,
        metadata_path:              str='',
        openpose_path:              str='',
    ) -> None:
        super().__init__(
            focal_length, principal_point, scale, blend, pad_scale, shoulder_scale,
            landmark_perc_threshold, joints2d, joints3d, j3d_head_index, mirror,
        )
        self.metadata_path = metadata_path
        self.openpose_paths = glob.glob(os.path.join(openpose_path, '*.json'))\
            if openpose_path and os.path.exists(openpose_path) else ''
        self.index = 0

    def create_metadata_path(self, index: int) -> str:
        md_fn = f"metadata_{index}.npz"
        if self.metadata_path and os.path.exists(self.metadata_path):
            md_fn = os.path.join(self.metadata_path, md_fn)
        return md_fn

    def create_openpose_path(self, index: int) -> str:
        return self.openpose_paths[index] if self.openpose_paths and os.path.exists(self.openpose_paths[index]) else ''

    def __call__(self, 
        tensors: typing.Dict[str, torch.Tensor],
        step:       typing.Optional[int]=None,
    ) -> None:
        b = tensors['joints2d'].shape[0]
        
        ret = super().__call__(tensors, [{
                'body': {
                    'image': f"image_{self.index + i}.png",
                    'overlay_t': f"overlay_{self.index + i}.jpg",
                    'padded_t': f"image_padded_{self.index + i}.png",
                    'body_legacy_t': f"body_legacy_{self.index + i}.pkl",
                    'body_t': f"body_{self.index + i}.pkl",
                    'metadata_t': self.create_metadata_path(self.index + i),
                    'openpose': self.create_openpose_path(self.index + i),
                }
            } for i in range(b)
        ])        
        self.index = self.index + b
        log.warning(f"head status: {ret[0]['message']} @ {self.index}")