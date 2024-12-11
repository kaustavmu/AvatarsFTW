from ts.utils.util import PredictionException
from scipy.spatial.transform import Rotation as R

import torch
import roma
import numpy as np
import logging
import os

log = logging.getLogger(__name__)

__all__ = ['HeadAngle']

def _get_global_poses(
    local_poses: torch.Tensor,
    parents: torch.Tensor, 
    output_format: str='aa', 
    input_format: str='aa'
) -> torch.Tensor:
    assert output_format in ['aa', 'rotmat']
    assert input_format in ['aa', 'rotmat']
    dof = 3 if input_format == 'aa' else 9
    n_joints = local_poses.shape[-1] // dof
    if input_format == 'aa':
        local_oris = roma.rotvec_to_rotmat(local_poses.reshape((-1, 3)))
    else:
        local_oris = local_poses
    local_oris = local_oris.reshape((-1, n_joints, 3, 3))
    global_oris = torch.zeros_like(local_oris)

    for j in range(n_joints):
        if parents[j] < 0: # root rotation            
            global_oris[..., j, :, :] = local_oris[..., j, :, :]
        else:
            parent_rot = global_oris[..., parents[j], :, :]
            local_rot = local_oris[..., j, :, :]
            global_oris[..., j, :, :] = torch.matmul(parent_rot, local_rot)

    if output_format == 'aa':
        global_oris = roma.rotmat_to_rotvec(global_oris.reshape((-1, 3, 3)))
        res = global_oris.reshape((-1, n_joints * 3))
    else:
        res = global_oris.reshape((-1, n_joints * 3 * 3))
    return res

class HeadAngle(torch.nn.Module):
    
    _SINGLE_ANGLE_THRESHOLD_ = 40.0
    _MULTI_ANGLE_THRESHOLD_ = 50.0
    
    def __init__(self, 
        # throw:  bool=True,
        single_angle_threshold:        float=40.0,
        multi_angle_threshold:         float=50.0,
    ):
        super().__init__()
        self.throw = bool(int(os.environ.get('KLOTHED_FIT_THROW_ON_ERROR', True)))
        self.single_angle_threshold = single_angle_threshold
        self.multi_angle_threshold = multi_angle_threshold
        self.register_buffer('parents', torch.Tensor([
            [
                -1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  
                8,  9,  9,  9, 12, 13, 14, 16, 17, 18, 19
            ],
            [
                0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 
                11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21
            ]
        ]).long())

    def forward(self, 
        local_body_pose:    torch.Tensor,
        global_rotation:    torch.Tensor,        
    ) -> torch.Tensor:
        global_pose = global_rotation.detach().cpu()
        body_pose = local_body_pose.detach().cpu()
        source_global_pose = _get_global_poses(
            torch.cat([global_pose, body_pose], dim=-1),
            self.parents.T[:, 0], output_format='aa', input_format='aa',
        )
        head_source_global_pose = source_global_pose[0, 15*3:16*3]
        X, Y, Z = R.from_matrix(
            roma.rotvec_to_rotmat(head_source_global_pose).numpy().squeeze()
        ).as_euler('xyz', degrees=True)
        X, Y, Z = np.abs(X), np.abs(Y), np.abs(Z)
        eX = X if X < 90.0 else 180.0 - X
        eY = Y if Y < 90.0 else 180.0 - Y
        eZ = Z if Z < 90.0 else 180.0 - Z
        eA = eX + eY + eZ
        
        error = eX >= self.single_angle_threshold or eY >= self.single_angle_threshold \
            or eZ >= self.single_angle_threshold or eA >= self.multi_angle_threshold
        if error:
            message = f"Head angle threshold exceeded [{eX:.0f}, {eY:.0f}, {eZ:.0f}] - {eA:.0f}"
            log.warning(message)
            if self.throw:
                raise PredictionException(message, 514)
            else:
                return torch.scalar_tensor(eA).float()
        else:
            return torch.scalar_tensor(eA).float()
        