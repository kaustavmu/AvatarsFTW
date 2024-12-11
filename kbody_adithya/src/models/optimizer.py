import torch
import pytorch_lightning
import omegaconf.omegaconf
import typing
import toolz
import os
import logging
import numpy as np

log = logging.getLogger(__name__)

__all__ = ['Optimizer']

__HAS_NV__ = os.environ.get('RENDER_BACKEND', 'nv') == 'nv'

log.warning(f"Using the {'nvdiffrast' if __HAS_NV__ else 'pytorch3d'} rendering backend.")

try:
    from optimization.variables import Shape, Pose
    from models.vposer import VPoser1
    from monads.joints import (
        JointConfidence, MergeToes, Regressors, JointMap, Split, JointRegressor
    )
    from models.smplx_body import SMPLX, Height, IPD
    from models.star_body import SMPLX2STAR, ToOpenPose
    from monads.virtual_joints import VirtualJoints
    from monads.camera import Camera
    from monads.rotation import Rotation3D, Rotate
    if __HAS_NV__:
        from monads.opengl import OpenGL
        from monads.silhouette import Silhouette
    else:        
        from monads.mesh_silhouette import Silhouette
    from monads.sampling import NearestSampling, BilinearSampling
    from monads.sobel import Sobel
    from losses.objective import Objective
    from losses.gm import GemanMcClure
    from losses.L2 import L2
    from losses.L1 import L1
    from losses.hinge import HingeJointPrior
    from metrics.rmse import RMSE
    from metrics.iou import IoU
    from metrics.head_angle import HeadAngle
    from metrics.fit_validation import FitValidator
except (ModuleNotFoundError, ImportError) as e:
    from variables import Shape, Pose
    from vposer import VPoser1
    from joints import (
        JointConfidence, MergeToes, Regressors, JointMap, Split, JointRegressor
    )
    from smplx_body import SMPLX, Height, IPD
    from star_body import SMPLX2STAR, ToOpenPose
    from virtual_joints import VirtualJoints
    from camera import Camera
    from rotation import Rotation3D, Rotate
    if __HAS_NV__:
        from opengl import OpenGL
        from silhouette import Silhouette
    else:
        from mesh_silhouette import Silhouette
    from sampling import NearestSampling, BilinearSampling
    from sobel import Sobel
    from objective import Objective
    from gm import GemanMcClure
    from L2 import L2
    from L1 import L1
    from hinge import HingeJointPrior
    from rmse import RMSE
    from iou import IoU
    from head_angle import HeadAngle
    from fit_validation import FitValidator

class Optimizer(pytorch_lightning.LightningModule):
    def __init__(self, 
        config: omegaconf.omegaconf.DictConfig,
    ):
        super(Optimizer, self).__init__()
        self.fwds = []
        self.predictions = { }
        self.setters = []        
        self.pose = Pose()
        self.shape = Shape()
        self.model = torch.nn.Identity()
        # self.initializers = [
        #     (i, _create_assigner(o)) for i, o in configuration.initialize.items()
        # ] if configuration.initialize is not None else []
        # init optimized params from loaded / predicted data (betas, latent, etc.)
        
        self.hole_w = config.objective.hole_w
        self.tol_g = config.objective.tol_g
        self.tol_e = config.objective.tol_e

        self.vposer = VPoser1()        
        vposer_ckpt = os.path.join(config.body.data_root, 'vposer_v1_0', 'snapshots', 'TR00_E096.pt')
        data = torch.load(vposer_ckpt, map_location=lambda s, l: s)                
        self.vposer.load_state_dict(data, strict=True)
        self.vposer = self.vposer.eval()
        # self.initializer = parameters.initialization if parameters is not None else None
        # init := vposer once, zero optimized params

        self.regressors = Regressors(**{
            'star': os.path.join(config.body.data_root, 'joints_regressors', 'star_body_regressor.npy'),
            'h36m': os.path.join(config.body.data_root, 'joints_regressors', 'smpl_J_regressor_h36m.npy'),
        })

        self.smplx = SMPLX(
            os.path.join(config.body.data_root, 'models_smplx_v1_1', 'models', 'smplx'),
            gender=config.body.gender, num_betas=config.body.num_betas,
            pca_components=12, use_translation=False, use_pose=False,
            use_face=False, use_hands=False, use_betas=False,
            use_global_orientation=False, use_eyes=True,
            flat_hand_mean=True, use_face_contour=True, joints_format=None,
        )

        self.smplx2star = SMPLX2STAR(
            os.path.join(config.body.data_root, 'star_1_1', 'def_transfer_smplx.npy')
        )

        self.preprocess = torch.nn.ModuleDict({
            'openpose_confidence_shoulders_feet_head_only': JointConfidence([0, 3, 4, 6, 7, 10, 13]),
            'star_ignore_pelvis_face_feet_neck': JointConfidence([0, 1, 2, 5, 8, 9, 12, 11, 22, 14, 19, 15, 16, 17, 18, 23, 24, 20, 21, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]), # /w hands
            'openpose_merge_toes': MergeToes(),            
        })
        # self.preprocess = _create_processing_block(feedforward, "preprocess", monads=monads)        

        self.postprocess = torch.nn.ModuleDict({
            'ipd': IPD(),
            'height': Height(),
            'openpose_jointmap': JointMap(with_hands=True, with_face=True, with_face_contour=True),
            'joint_regressor': JointRegressor(),
            'virtual_joints': VirtualJoints(**config.body.virtual_joints),
            'openpose_star_to_openpose': ToOpenPose(neck_indices=[13, 14]),
            'openpose_star_joint_map': JointMap(model='star', with_hands=True, with_face=True, with_face_contour=False),
            'openpose_star_split': Split('coco25_star+'),
            'weak_perspective_camera': Camera(),
            'openpose_split': Split('coco25_face'),
            'rotation3d': Rotation3D(['180@y', '180@z']),
            'rotate': Rotate(),
            'opengl': OpenGL(principal_point=None, width=180, height=320, persistent=True)\
                if __HAS_NV__ else torch.nn.Identity(),
            'mesh_silhouette': Silhouette(width=180, height=320),
            'nearest_sampling': NearestSampling(width=180, height=320),
            'bilinear_sampling': BilinearSampling(width=180, height=320),
            'sobel': Sobel(),
            # 'scale': * HOLE_W(10.0),
            # 'multiply': torch.mul,
        })
        # self.postprocess = _create_processing_block(feedforward, "postprocess", monads=monads)
        
        self.losses = torch.nn.ModuleDict({
            'gm': GemanMcClure(),
            'L2': L2(),
            'L1': L1(),
            'hinge': HingeJointPrior()
        })

        # self.optimizer_configs = toolz.get_in(['optimization', 'optimizers'], parameters) or []
        # optimizer hparams

        # self.parameter_selectors = parameters.selectors or []
        # select optimized params (i.e. the named params)

        self.supervision = torch.nn.ModuleDict()
        self.params_optimizers = {}
        self.stages = []
        
        # optimization_process = toolz.get_in(['optimization', 'process'], parameters) or { }
        
        log.info(f"Optimization with {len(config.objective.process)} stages:")
        for i, (name, stage) in enumerate(config.objective.process.items()):
            log.info(f"\t stage#{i}: {name}#{stage.iterations}")
            self.stages.append(name)
            self.supervision[name] = Objective(stage.weights)
            self.params_optimizers[name] = {
                'iterations': int(stage.iterations),
                'disentangled': bool(stage.disentangled),
                'parameters': stage.parameters
            }
        # for stage, cfg in optimization_process.items():
        #     self.stages.append(stage)
        #     log.info(
        #         f"Setting up the '{stage}' stage using the " + str(cfg.optimizer or "same") + " optimizer, "
        #         f"optimizing the " + str(cfg.selectors or "same") + " parameters"
        #     )
        #     optimizer = cfg.optimizer
        #     iterations = cfg.iterations
        #     scheduler = cfg.scheduler
        #     selector = cfg.selectors
        #     self.params_optimizers.append((optimizer, stage, iterations, selector, scheduler))
        #     objective = cfg.objective #TODO: merge confs from supervision and objectives
        #     self.supervision[stage] = _create_supervision_block(
        #         omegaconf.OmegaConf.merge(supervision, objective)
        #     )

        self.metrics = torch.nn.ModuleDict({
            'rmse': RMSE(),
            'iou': IoU(),
            'fit': FitValidator(),
            'head': HeadAngle(),
        })
        # self.validation = _create_validation_block(validation) #TODO: change this, "empty processing block" is confusing
        # # rmse, iou, validate fit, head angle rejection
        
        self.optimization_step = 0
        self.prediction_stages = []

    def initialize_parameters(self) -> None:
        # init = hyu.instantiate(self.initializer) if self.initializer else NoInit()
        # init(self)
        pass

    # def initialize(self,
    #     tensors: typing.Dict[str, torch.Tensor]
    # ) -> None:
    #     for i, a in self.initializers:
    #         accessor = _create_accessor(i)
    #         a(self,accessor(tensors))
    #         #a(self, tensors[i])

    # def assign(self,
    #     tensors: typing.Dict[str, torch.Tensor]
    # ) -> None:
    #     for i, a in self.assigners:
    #         a(self, tensors[i])

    # def forward(self,
    #     tensors: typing.Dict[str, torch.Tensor]
    # ) -> typing.Dict[str, torch.Tensor]:
    #     for f in self.fwds:
    #         f(tensors)
    #     for k in self.setters:
    #         self.predictions[k] = tensors[k]
    #     return tensors
    
    def assign_predictions_to_params(self, predictions: dict) -> None:
        self.pose.expression_t.copy_(predictions['expression'].clone())
        self.pose.global_orient_t.copy_(predictions['global_orient'].clone())
        self.pose.translation_t.copy_(predictions['pnp_translation'].clone())
        # pose.translation_t.copy_(predictions['perspective_translation'].clone())
        self.pose.pose_t.copy_(predictions['latent_pose'].clone())
        self.pose.jaw_t.copy_(predictions['jaw_pose'].clone())
        self.pose.lhand_t.data.zero_()
        self.pose.rhand_t.data.zero_()
        b = predictions['betas']
        for e, p in zip(b[0], self.shape.parameters()):
            p.data = e[np.newaxis, np.newaxis].clone()

    def get_optimized_params(self, mapping: typing.Mapping[str, typing.Sequence]):
        params = []
        if 'pose' in mapping:
            for k in mapping['pose']:
                params.append(getattr(self.pose, k))
        if 'shape' in mapping:
            for i in range(mapping['shape'][0], mapping['shape'][1]):
                params.append(getattr(self.shape, f"{i}"))
        return params

    def training_step(self, 
        batch:                  typing.Dict[str, torch.Tensor],
        batch_idx:              int,
        optimizer_idx:          int=0,
    ) -> typing.Dict[str, typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]]:
        # td = self.preprocess(batch)
        batch['betas_t'] = self.shape.forward()
        batch['params'] = self.pose.forward()
        batch['smplx_confidence'] = self.preprocess['openpose_confidence_shoulders_feet_head_only'](
            confidence=batch['confidence']
        )
        batch['star_confidence'] = self.preprocess['star_ignore_pelvis_face_feet_neck'](
            confidence=batch['confidence']
        )
        batch['star_openpose'] = self.preprocess['openpose_merge_toes'](
            keypoints=batch['keypoints'],
            confidence=batch['star_confidence']
        )
        batch['regressors'] = self.regressors()

        objective = self.supervision[self.stages[optimizer_idx]]
        
        # td = self.postprocess(batch)
        batch['decoded'] = self.vposer(decode=batch['params']['pose_t'])
        batch['body'] = self.smplx(
            shape=batch['betas_t'],
            pose=batch['decoded']['pose'],
            rotation=batch['params']['global_orient_t'],
            translation=None,
            left_hand=batch['params']['lhand_t'],
            right_hand=batch['params']['rhand_t'],
            expression=batch['params']['expression_t'],
            jaw=batch['params']['jaw_t'],
            left_eye=None,
            right_eye=None,
        )
        batch['ipd'] = self.postprocess['ipd'](batch['body']['joints'])
        batch['height'] = self.postprocess['height'](batch['body']['shape'])
        batch['star_vertices'] = self.smplx2star(batch['body']['vertices'])
        batch['smplx_joints'] = self.postprocess['openpose_jointmap'](batch['body']['joints'])
        batch['body_joints'] = self.postprocess['joint_regressor'](
            batch['star_vertices'], batch['regressors']['star']
        )
        batch['pelvis_joints'] = self.postprocess['joint_regressor'](
            batch['star_vertices'], batch['regressors']['h36m']
        )
        batch['smplx_joints'] = self.postprocess['virtual_joints'](
            raw=batch['body']['joints'],
            j14=batch['pelvis_joints'],
            openpose=batch['smplx_joints'],
        )
        batch['body_joints'] = self.postprocess['openpose_star_to_openpose'](
            joints=batch['body_joints'], vertices=batch['star_vertices']
        )
        batch['body_joints'] = self.postprocess['openpose_star_joint_map'](
            batch['body_joints']
        )
        batch['split_body_joints'] = self.postprocess['openpose_star_split'](
            batch['body_joints']
        )
        batch['joints2d'] = self.postprocess['weak_perspective_camera'](
            points=batch['smplx_joints'], image=batch['color'],
            translation=batch['params']['translation_t'],
            rotation=batch['camera_rotation'],
            intrinsics=batch['camera_intrinsics'],
        )
        batch['body_joints2d'] = self.postprocess['weak_perspective_camera'](
            points=batch['split_body_joints']['body'], image=batch['color'],
            translation=batch['params']['translation_t'],
            rotation=batch['camera_rotation'],
            intrinsics=batch['camera_intrinsics'],
        )
        for i, o in zip([
            'joints2d', 'keypoints', 'star_confidence', 'smplx_confidence'
        ], [
            'split_joints2d', 'split_smplx_wrists_feet_head', 'split_star_confidence', 'split_smplx_confidence'
        ]):
            batch[o] = self.postprocess['openpose_split'](batch[i])
        batch['split_keypoints'] = self.postprocess['openpose_split'](batch['star_openpose']['positions'])
        batch['split_confidence'] = self.postprocess['openpose_split'](batch['star_openpose']['confidence'])
        if objective.requires_rendering():
            batch['rotation'] = self.postprocess['rotation3d'](batch['body']['vertices'])
            batch['rotated_vertices'] = self.postprocess['rotate'](
                rotation=batch['rotation'], points=batch['body']['vertices']
            )
            batch['rotated_vertices'] = self.postprocess['rotate'](
                rotation=batch['camera_rotation'], points=batch['rotated_vertices']
            )
            if __HAS_NV__:
                batch['ndc_vertices'] = self.postprocess['opengl'](
                    points=batch['rotated_vertices'],
                    translation=batch['params']['translation_t'],
                    rotation=None,
                    nominal_image=batch['color'],
                    intrinsics=batch['camera_intrinsics'],
                )
                batch['silhouette_t'] = self.postprocess['mesh_silhouette'](
                    ndc_vertices=batch['ndc_vertices'], indices=batch['body']['faces'],
                )
            else:
                batch['silhouette_t'] = self.postprocess['mesh_silhouette'](
                    vertices=batch['rotated_vertices'], faces=batch['body']['faces'],
                    rotation=None, translation=batch['params']['translation_t'],
                    nominal_image=batch['color'],
                )
            batch['holes_down'] = self.postprocess['nearest_sampling'](image=batch['holes'])
            batch['color_down_x2'] = self.postprocess['bilinear_sampling'](image=batch['color'])
            batch['silhouette_down_x2'] = self.postprocess['bilinear_sampling'](image=batch['silhouette'])
            batch['edt_down_x2'] = self.postprocess['bilinear_sampling'](image=batch['edt'])
            batch['edges_t'] = self.postprocess['sobel'](image=batch['silhouette_t'])
            batch['holes_down'] *= self.hole_w
            batch['chamfer'] = batch['edges_t'] * batch['edt_down_x2']
            batch['chamfer_holes'] = batch['chamfer'] * batch['holes_down']

        total_loss, losses = objective(batch, self.losses)
        self.optimization_step += 1
        return { 'loss': total_loss, 'losses': losses, 'tensors': batch }

    def validation(self, 
        tensors: typing.Mapping[str, typing.Union[torch.Tensor, typing.Any]]
    ) -> typing.Mapping[str, torch.Tensor]:
        return {
            'rmse': self.metrics['rmse'](
                gt=tensors['split_keypoints']['body'],
                pred=tensors['body_joints2d'],
                weights=tensors['split_star_confidence']['body'],
            ),
            'iou': self.metrics['iou'](
                gt=tensors.get('silhouette_down_x2', None),
                pred=tensors.get('silhouette_t', None),
            ),
            'valid': self.metrics['fit'](
                translation=tensors['params']['translation_t'],
                rotation=tensors['params']['global_orient_t'],
                betas=tensors['betas_t'],
            ),
            'head_angle': self.metrics['head'](
                local_body_pose=tensors['decoded']['pose'],
                global_rotation=tensors['params']['global_orient_t'],
            ),
        }

    def configure_optimizers(self) -> typing.Tuple[typing.List[torch.optim.Optimizer], typing.List[torch.optim.lr_scheduler._LRScheduler]]:
        log.info(f"Configuring optimizers")
        optimizers = []
        for stage in self.stages:
            params = self.params_optimizers[stage]
            if params['disentangled']:
                pose_params = self.get_optimized_params(toolz.dissoc(params['parameters'], 'shape'))
                shape_params = self.get_optimized_params(toolz.dissoc(params['parameters'], 'pose'))
                optimizers.append(AlternatingOptimizer([
                    torch.optim.LBFGS(
                        shape_params, # pose_params,                 
                        tolerance_change=self.tol_e, tolerance_grad=self.tol_g,
                        lr=1.0, max_iter=100, line_search_fn='strong_wolfe'
                    ),
                    torch.optim.LBFGS(
                        pose_params, # shape_params,                 
                        tolerance_change=self.tol_e, tolerance_grad=self.tol_g,
                        lr=1.0, max_iter=30, line_search_fn='strong_wolfe'
                    ),                    
                ], shape_params + pose_params))
            else:
                optimizers.append(torch.optim.LBFGS(
                    self.get_optimized_params(params['parameters']),                 
                    tolerance_change=self.tol_e, tolerance_grad=self.tol_g,
                    lr=1.0, max_iter=30, line_search_fn='strong_wolfe'
                ))
            optimizers[-1].name = [stage]
            optimizers[-1].iterations = [params['iterations']]
        return optimizers
        
        # optimizers, schedulers = [], []        
        # for optimizer, name, iterations, params, schedule in self.params_optimizers:
        #     if optimizer is None and params is None:
        #         optimizers.append(optimizers[-1])
        #         schedulers.append(schedulers[-1])
        #         getattr(optimizers[-1], 'iterations').append(iterations)
        #         getattr(optimizers[-1], 'name').append(name)
        #     elif isinstance(optimizer, str):
        #         parameters = hyu.instantiate(self.parameter_selectors[params])(self) if isinstance(params, str) else\
        #             list(hyu.instantiate(self.parameter_selectors[p])(self) for p in params)
        #             # list(toolz.concat(hyu.instantiate(self.parameter_selectors[p])(self) for p in params))                    
        #         #TODO: parameter construction is very slow
        #         optimizers.append(_create_optimization_block(
        #             self.optimizer_configs[optimizer], parameters
        #         ))#TODO: check if it works with a list of parameters
        #         setattr(optimizers[-1], 'iterations', [iterations])
        #         setattr(optimizers[-1], 'name', [name])
        #         schedulers.append(_create_scheduling_block(
        #             self.scheduler_configs.get(schedule, None), [optimizers[-1]]
        #         ))
        #         if any(p in (self.assigned_params or []) for p in params):
        #             optimizers[-1].assign = self.assign
        #             self.prediction_stages.append(name)
        #     else:
        #         parameters = [
        #             hyu.instantiate(self.parameter_selectors[par])(self) if isinstance(par, str)
        #             # else list(toolz.concat(hyu.instantiate(self.parameter_selectors[p])(self) for p in par))
        #             else list(hyu.instantiate(self.parameter_selectors[p])(self) for p in par)
        #             for par in params
        #         ]
        #         alternated = [_create_optimization_block(
        #                 self.optimizer_configs[o], param  
        #             ) for o, param in zip(optimizer, parameters)
        #         ]
        #         for ap in self.assigned_params or []:
        #             alternated[params.index(ap)].assign = self.assign
        #             self.prediction_stages.append(name)
        #         if isinstance(toolz.first(parameters), dict) and len(parameters) > 1:
        #             parameters = toolz.mapcat(lambda d: d['params'], parameters)
        #         optimizers.append(AlternatingOptimizer(
        #             alternated, list(parameters)
        #             # alternated, list(toolz.concat(parameters))
        #         ))
        #         setattr(optimizers[-1], 'iterations', [iterations])
        #         setattr(optimizers[-1], 'name', [name])                
        #         schedulers.append(_create_scheduling_block(
        #             self.scheduler_configs.get(schedule, None), [optimizers[-1]]
        #         ))
        # return (
        #     optimizers,
        #     list(map(lambda s: s.schedulers[0] if isinstance(s, NoScheduling) else s, schedulers))
        # )

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: typing.Union[torch.optim.Optimizer, pytorch_lightning.core.optimizer.LightningOptimizer],
        optimizer_idx: int = 0,
        optimizer_closure: typing.Optional[typing.Callable[[], typing.Any]] = None,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ) -> None:        
        optimizer.step(closure=optimizer_closure)    

class AlternatingOptimizer(torch.optim.Optimizer):
    def __init__(self, 
        optimizers: typing.Iterable[torch.optim.Optimizer],
        parameters: typing.Iterable[torch.nn.parameter.Parameter],
    ):
        self.optimizers = optimizers
        super(AlternatingOptimizer, self).__init__(
            parameters, {'lr': 1.0}
        )

    def step(self, closure):
        for o in self.optimizers:
            o.step(closure)