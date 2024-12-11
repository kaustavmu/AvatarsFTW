from src.handlers.output import KlothedBodyV3

import os
import glob
import torch
import typing
import logging

log = logging.getLogger(__name__)

__all__ = ['KlothedV3']

class KlothedV3(KlothedBodyV3):
    def __init__(self,
        focal_length:               typing.Union[float, typing.Tuple[float, float]]=5000.0,
        principal_point:            typing.Optional[typing.Union[float, typing.Tuple[float, float]]]=None,
        scale:                      float=1.0,
        blend:                      float=0.65,
        joints3d:                   str='smplx_joints',
        j3d_head_index:             int=0,
        metadata_path:              str='',
        openpose_path:              str='',
        matte_path:                 str='',
        has_decomposed_betas:       bool=False,
        gender:                     str='neutral',
        height_regressors:          str='',
    ) -> None:
        super().__init__(
            focal_length, principal_point, scale, blend, joints3d, j3d_head_index,
            has_decomposed_betas, gender, height_regressors,
        )        
        self.metadata_path = metadata_path
        self.openpose_paths = glob.glob(os.path.join(openpose_path, '*.json'))\
            if openpose_path and os.path.exists(openpose_path) else ''
        self.matte_paths = glob.glob(os.path.join(matte_path, '*_silhouette.jp*g'))\
            if matte_path and os.path.exists(matte_path) else ''
        self.index = 0

    def create_metadata_path(self, index: int) -> str:
        md_fn = f"metadata_{index:05d}.npz"
        if self.metadata_path and os.path.exists(self.metadata_path):
            md_fn = os.path.join(self.metadata_path, md_fn)
        return md_fn

    def create_openpose_path(self, index: int) -> str:
        return self.openpose_paths[index] if self.openpose_paths and os.path.exists(self.openpose_paths[index]) else ''
    
    def create_matte_path(self, index: int) -> str:
        return self.matte_paths[index] if self.matte_paths and os.path.exists(self.matte_paths[index]) else ''

    def __call__(self, 
        tensors: typing.Dict[str, torch.Tensor],
        step:       typing.Optional[int]=None,
    ) -> None:
        b = tensors['joints2d'].shape[0]
        
        ret = super().__call__(tensors, [{
                'body': {
                    'image': f"image_{self.index + i}.png",
                    'overlay_t': f"overlay_{self.index + i:05d}.jpg",
                    'padded_t': f"image_padded_{self.index + i:05d}.png",
                    'body_legacy_t': f"body_legacy_{self.index + i:05d}.pkl",
                    'body_t': f"body_{self.index + i:05d}.pkl",
                    'metadata_t': self.create_metadata_path(self.index + i),
                    'keypoints': self.create_openpose_path(self.index + i),
                    'matte': self.create_matte_path(self.index + i),
                    'matte_t': f"matte_padded_{self.index + i:05d}.exr",
                    "betas_t": f"betas_{self.index + i:05d}.txt",
                }
            } for i in range(b)
        ])        
        self.index = self.index + b
        log.warning(f"head status: {ret[0]['message']} @ {self.index}")