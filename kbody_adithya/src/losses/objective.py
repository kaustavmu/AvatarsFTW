import torch
import typing
import logging
import toolz

log = logging.getLogger(__name__)

__all__ = ['Objective']

class Objective(torch.nn.Module):
    def __init__(self, weights: typing.Mapping[str, typing.Any]) -> None:
        super().__init__()
        self.weights = toolz.valmap(lambda v: float(v), weights)

    def requires_rendering(self,) -> bool:
        return self.weights.get('data_edt', None) or self.weights.get('data_mask', None)

    def forward(self, 
            tensors: typing.Dict[str, torch.Tensor],
            losses: torch.nn.ModuleDict,
        ) -> torch.Tensor:
        device = next(toolz.take(1, 
            filter(lambda t: isinstance(t, torch.Tensor), tensors.values())
        )).device
        error = torch.tensor(0.0, dtype=torch.float32, device=device)
        errors = {}
        if self.weights.get('data_body', 0.0):
            gm_body = losses['gm'](
                gt=tensors['split_keypoints']['body'],
                pred=tensors['body_joints2d'],
                weights=tensors['split_star_confidence']['body'],
            ).sum() + losses['gm'](
                gt=tensors['split_smplx_wrists_feet_head']['body'],
                pred=tensors['split_joints2d']['body'],
                weights=tensors['split_smplx_confidence']['body'],
            ).sum()
            error += self.weights['data_body'] * gm_body
            errors['gm_body'] = gm_body
        if self.weights.get('data_hands', 0.0):
            gm_hands = losses['gm'](
                gt=tensors['split_keypoints']['hands'],
                pred=tensors['split_joints2d']['hands'],
                weights=tensors['split_smplx_confidence']['hands'],
            ).sum()
            error += self.weights['data_hands'] * gm_hands
            errors['gm_hands'] = gm_hands
        if self.weights.get('data_face', 0.0):
            gm_face = losses['gm'](
                gt=tensors['split_keypoints']['face'],
                pred=tensors['split_joints2d']['face'],
                weights=tensors['split_smplx_confidence']['face'],
            ).sum()
            error += self.weights['data_face'] * gm_face
            errors['gm_face'] = gm_face
        if self.weights.get('prior_shape', 0.0):
            L2_shape = losses['L2'](
                pred=tensors['betas_t'],
            ).sum()
            error += self.weights['prior_shape'] * L2_shape
            errors['L2_shape'] = L2_shape
        if self.weights.get('prior_pose', 0.0):
            L2_pose = losses['L2'](
                pred=tensors['params']['pose_t'],
            ).sum()
            error += self.weights['prior_pose'] * L2_pose
            errors['L2_pose'] = L2_pose
        if self.weights.get('prior_expression', 0.0):
            L2_expression = losses['L2'](
                pred=tensors['params']['expression_t'],
            ).sum()
            error += self.weights['prior_expression'] * L2_expression
            errors['L2_expression'] = L2_expression
        if self.weights.get('prior_hinge', 0.0):
            hinge = losses['hinge'](
                pose=tensors['decoded']['pose'],
            ).mean() # .sum()
            error += self.weights['prior_hinge'] * hinge
            errors['hinge'] = hinge
        if self.weights.get('data_mask', 0.0):
            mask = losses['L1'](
                gt=tensors['silhouette_down_x2'],
                pred=tensors['silhouette_t'],
                weights=None
            ).sum()
            error += self.weights['data_mask'] * mask
            errors['mask'] = mask
        if self.weights.get('data_mask_holes', 0.0):
            mask_holes = losses['L1'](
                gt=tensors['silhouette_down_x2'],
                pred=tensors['silhouette_t'],
                weights=tensors['holes_down'],
            ).sum()
            error += self.weights['data_mask_holes'] * mask_holes
            errors['mask_holes'] = mask_holes
        if self.weights.get('data_edt', 0.0):
            chamfer = tensors['chamfer'].sum()
            error += self.weights['data_edt'] * chamfer
            errors['chamfer'] = chamfer
            # print(f"chamfer={chamfer} -- {tensors['params']['translation_t']}")
        if self.weights.get('data_edt_holes', 0.0):
            chamfer_holes = tensors['chamfer_holes'].sum()
            error += self.weights['data_edt_holes'] * chamfer_holes
            errors['chamfer_holes'] = chamfer_holes
        return error, errors