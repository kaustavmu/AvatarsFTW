from collections.abc import Callable

import typing
import torch
import logging
import pickle
import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    'Body',
    'BodyLegacy',
]

class Body(Callable):
    def __init__(self,
        
    ) -> None:
        super().__init__()

    def __call__(self, 
        data:   typing.Mapping[str, torch.Tensor],
        json:   typing.Sequence[typing.Mapping[str, typing.Any]],
    ) -> typing.Dict[str, torch.Tensor]:
        outs = []
        for i in range(len(json)):
            with open(json[i]['body']['body_t'], 'wb') as output_file:
                pickle.dump({                    
                    'camera_rotation': np.eye(3, dtype=np.float32),
                    'leye_pose': np.zeros(3, dtype=np.float32),
                    'reye_pose': np.zeros(3, dtype=np.float32),
                    'global_orient': data['params']['global_orient_t'][i].detach().cpu().numpy(),
                    'camera_translation': data['params']['translation_t'][i].detach().cpu().numpy(),
                    'latent_pose': data['params']['pose_t'][i].detach().cpu().numpy(),
                    'betas': data['params']['betas_t'][i].detach().cpu().numpy(),
                    'jaw_pose': data['params']['jaw_t'][i].detach().cpu().numpy(),
                    'right_hand_pose': data['params']['lhand_t'][i].detach().cpu().numpy(),
                    'left_hand_pose': data['params']['rhand_t'][i].detach().cpu().numpy(),
                    'expression': data['params']['expression_t'][i].detach().cpu().numpy(),
                    'body_pose': data['body']['pose'][i].detach().cpu().numpy(),
                }, output_file)
            outs.append({ 'body_t': 'Success' })
        return outs


class BodyLegacy(Callable):
    def __init__(self,
        
    ) -> None:
        super().__init__()

    def __call__(self, 
        data:   typing.Mapping[str, torch.Tensor],
        json:   typing.Sequence[typing.Mapping[str, typing.Any]],
    ) -> typing.Dict[str, torch.Tensor]:
        outs = []
        for i in range(len(json)):
            with open(json[i]['body']['body_legacy_t'], 'wb') as output_file:
                pickle.dump({                    
                    'camera_rotation': np.eye(3, dtype=np.float32)[np.newaxis, ...],
                    'leye_pose': np.zeros((1, 3), dtype=np.float32),
                    'reye_pose': np.zeros((1, 3), dtype=np.float32),
                    'global_orient': data['params']['global_orient_t'][i][np.newaxis, ...].detach().cpu().numpy(),
                    'camera_translation': data['params']['translation_t'][i][np.newaxis, ...].detach().cpu().numpy(),
                    'betas': data['params']['betas_t'][i][np.newaxis, ...].detach().cpu().numpy(),
                    'jaw_pose': data['params']['jaw_t'][i][np.newaxis, ...].detach().cpu().numpy(),
                    'right_hand_pose': data['params']['lhand_t'][i][np.newaxis, ...].detach().cpu().numpy(),
                    'left_hand_pose': data['params']['rhand_t'][i][np.newaxis, ...].detach().cpu().numpy(),
                    'expression': data['params']['expression_t'][i][np.newaxis, ...].detach().cpu().numpy(),
                    'body_pose': data['params']['pose_t'][i][np.newaxis, ...].detach().cpu().numpy(),
                }, output_file)
            outs.append({ 'body_legacy_t': 'Success' })
        return outs