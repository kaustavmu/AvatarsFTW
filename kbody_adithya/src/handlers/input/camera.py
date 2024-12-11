from collections.abc import Callable
from scipy.spatial.transform import Rotation as R

import numpy as np
import typing
import torch
import logging
import pickle
import os

log = logging.getLogger(__name__)

__all__ = ['Camera']

def load_pkl_file(
    filename:   str,
) -> typing.Dict[str, torch.Tensor]:
    data = { }
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

class Camera(Callable):
    def __init__(self,
        force_identity:     bool=False,
    ) -> None:
        super().__init__()
        self.force_identity = force_identity

    def __call__(self, 
        data:   typing.Mapping[str, typing.Any],
        device: torch.device,
    ) -> typing.Dict[str, torch.Tensor]:
        filename = data.get('camera', '')
        if not self.force_identity and os.path.exists(filename):
            log.info(f"Loading camera parameters file [key: body] @ {filename}")
            data = load_pkl_file(filename)
            log.info(f"Using a camera with a pitch rotation of {data['pitch']} degrees.")
            rotation = R.from_euler('x', data['pitch'], degrees=True)
            # return { 'camera_rotation': torch.from_numpy(rotation.as_matrix()).float().to(device) }
            return { 'camera_rotation': torch.from_numpy(rotation.as_matrix()).float().inverse()[np.newaxis, ...].to(device) }
            # log.warning(f"Camera pitch is disabled, proceeding with an identity camera rotation.")
            # return { 'camera_rotation': torch.eye(3).float()[np.newaxis, ...].to(device) }
        else:
            submsg = "ignored" if self.force_identity else "does not exist"
            log.warning(f"Camera parameters file ({filename}) {submsg}, proceeding with an identity camera rotation.")
            return { 'camera_rotation': torch.eye(3).float()[np.newaxis, ...].to(device) }