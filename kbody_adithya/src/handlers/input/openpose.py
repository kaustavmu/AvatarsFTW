from collections.abc import Callable

import typing
import torch
import json
import numpy as np
import logging

log = logging.getLogger(__name__)

__all__ = ['OpenPoseFile']

def _get_area(keypoints: torch.Tensor) -> float:
    min_x = keypoints[..., 0].min()
    min_y = keypoints[..., 1].min()
    max_x = keypoints[..., 0].max()
    max_y = keypoints[..., 1].max()
    return (max_x - min_x) * (max_y - min_y) * keypoints[..., 2].sum()

class OpenPoseFile(Callable):
    def __init__(self,
        load_face_contour:  bool=False,
    ) -> None:
        super().__init__()
        self.load_face_contour = load_face_contour

    def __call__(self, 
        data:   typing.Mapping[str, typing.Any],
        device: torch.device,
    ) -> typing.Dict[str, torch.Tensor]:
        filename = data.get('keypoints')
        log.info(f"Loading OpenPose keypoints file [key: keypoints] @ {filename}")
        with open(filename) as keypoint_file:
            json_data = json.load(keypoint_file)
        keypoints = []
        for person in json_data['people']:
            body = np.array(person['pose_keypoints_2d'], dtype=np.float32)
            body = body.reshape([-1, 3])
            left_hand = np.array(person['hand_left_keypoints_2d'], dtype=np.float32).reshape([-1, 3])
            right_hand = np.array(person['hand_right_keypoints_2d'], dtype=np.float32).reshape([-1, 3])
            body = np.concatenate([body, left_hand, right_hand], axis=0)
            face = np.array(person['face_keypoints_2d'], dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]
            contour_keyps = np.array([], dtype=body.dtype).reshape(0, 3)
            if self.load_face_contour:
                contour_keyps = np.array(person['face_keypoints_2d'], dtype=np.float32).reshape([-1, 3])[:17, :]
            body = np.concatenate([body, face, contour_keyps], axis=0)
            keypoints.append(torch.from_numpy(body))
        keypoints = [max(keypoints, key=_get_area)]
        keypoints = torch.stack(keypoints, dim=0).squeeze()
        return {
            'keypoints': keypoints[np.newaxis, ..., :2].to(device),
            'confidence': keypoints[np.newaxis, ..., 2:3].to(device),
        }