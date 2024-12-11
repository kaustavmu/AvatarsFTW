import typing
import torch
import logging

__all__ = ['VirtualJoints']

log = logging.getLogger(__name__)

class VirtualJoints(torch.nn.Module):
    def __init__(self,
        neck_uv:        float,
        pelvis_uv:      float,
        shoulder_uv:    typing.Mapping[str, float],
        hip_uv:         typing.Mapping[str, float],
        persistent:     bool=True,
    ) -> None:
        super().__init__()
        self.register_buffer('neck_weights', torch.tensor(
            [1.0 - 2.0 * neck_uv, neck_uv, neck_uv]).unsqueeze(-1), 
            persistent=persistent
        )
        self.register_buffer('pelvis_weights', torch.tensor(
            [1.0 - 2.0 * pelvis_uv, pelvis_uv, pelvis_uv]).unsqueeze(-1), 
            persistent=persistent
        )
        self.register_buffer('shoulder_weights', torch.tensor([
                1.0 - shoulder_uv['neck'] - shoulder_uv['clavicle'], 
                shoulder_uv['neck'], 
                shoulder_uv['clavicle']
            ]).unsqueeze(-1), 
            persistent=persistent
        )
        self.register_buffer('lshoulder_indices', 
            torch.tensor([16, 12, 13]).long(),
            persistent=persistent
        )
        self.register_buffer('rshoulder_indices', 
            torch.tensor([17, 12, 14]).long(),
            persistent=persistent
        )
        self.register_buffer('hip_weights', torch.tensor([                
                hip_uv['hip'],
                hip_uv['pelvis'],                 
                1.0 - hip_uv['pelvis'] - hip_uv['hip'], 
            ]).unsqueeze(-1), 
            persistent=persistent
        )
        log.info(f"Using virtual joints parameterized as: neck({neck_uv}) - pelvis({pelvis_uv}) - shoulders({shoulder_uv['neck']} | {shoulder_uv['clavicle']}) - hips({hip_uv['hip']} | {hip_uv['pelvis']})")
        # for hips, pelvis is @ 0 and hips @ [1, 2] for left/right for SMPLX
        # and crest comes from the j14 regressor @ [3, 2] for left/right

    def forward(self,
        raw:        torch.Tensor,
        j14:        torch.Tensor,
        openpose:   torch.Tensor,
    ) -> torch.Tensor:
        # pelvis @ 0, hips @ [1, 2] for SMPLX, pelvis @ 8 for OpenPose
        openpose[:, 8, :] = torch.sum(raw[:, :3, :] * self.pelvis_weights, dim=1)
        # neck @ 12, clavicles @ [13, 14] for SMPLX, neck @ 1 for OpenPose
        openpose[:, 1, :] = torch.sum(raw[:, 12:15, :] * self.neck_weights, dim=1)
        # lshoulder @ 16, neck & lclavicle @ [12, 13] for SMPLX, lshoulder @ 5 for OpenPose
        openpose[:, 5, :] = torch.sum(
            torch.index_select(raw, dim=1, index=self.lshoulder_indices) * self.shoulder_weights,
            dim=1
        )
        # rshoulder @ 17, neck & lclavicle @ [12, 14] for SMPLX, rshoulder @ 2 for OpenPose
        openpose[:, 2, :] = torch.sum(
            torch.index_select(raw, dim=1, index=self.rshoulder_indices) * self.shoulder_weights,
            dim=1
        )
        # lhip @ 12 for OpenPose, pelvis @ 0, lhip @ 1 for SMPLX, and lcrest @ 1 for h36m OR 3 for SMPLX2J14
        openpose[:, 12, :] = torch.sum(
            # torch.stack([raw[:, 1, :], raw[:, 0, :], j14[:, 3, :]], dim=1) * self.hip_weights,
            torch.stack([raw[:, 1, :], raw[:, 0, :], j14[:, 1, :]], dim=1) * self.hip_weights,
            dim=1
        )
        # rhip @ 9 for OpenPose, pelvis @ 0, rhip @ 2 for SMPLX, and rcrest @ 4 for h36m OR 2 for SMPLX2J14
        openpose[:, 9, :] = torch.sum(
            # torch.stack([raw[:, 2, :], raw[:, 0, :], j14[:, 2, :]], dim=1) * self.hip_weights,
            torch.stack([raw[:, 2, :], raw[:, 0, :], j14[:, 4, :]], dim=1) * self.hip_weights,
            dim=1
        )
        return openpose
