from scipy.spatial.transform import Rotation as R

import torch
import glob
import os
import typing
import logging
import pickle

__all__ = ["CamCalib"]

log = logging.getLogger(__name__)

class CamCalib(torch.utils.data.Dataset):
    def __init__(self,
        root:           str='',
    ):
        self.files = glob.glob(os.path.join(root, '*.pkl'))
        log.info(f"Loaded {len(self)} .pkl files.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:        
        with open(self.files[index], 'rb') as f:
            data = pickle.load(f)
        log.info(f"Using a camera with a pitch rotation of {data['pitch']} degrees.")
        rotation = R.from_euler('x', data['pitch'], degrees=True)
        # return { 'camera_rotation': torch.from_numpy(rotation.as_matrix()).float() }
        # return { 'camera_rotation': torch.from_numpy(rotation.as_matrix()).float().inverse() }
        return { 'camera_rotation': torch.eye(3).float() }
        