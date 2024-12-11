from ts.torch_handler.base_handler import BaseHandler
from omegaconf.omegaconf import OmegaConf

import toolz
import os
import torch
import logging
import typing

try:
    from models.optimizer import Optimizer
    from engine.init_cuda_error_fix import InitCudaErrorFix
    from engine.seed import ManualSeed
    from engine.cudnn import DisableCuDNN, DisableTensorCores
    from handlers.input.image import ImageFileInput
    from handlers.input.body import Body
    from handlers.input.metadata import Metadata
    from handlers.input.openpose import OpenPoseFile
    from handlers.input.silhouette_holes import SilhouetteHoles
    from handlers.output.klothed_v4_1 import KlothedBodyV4_1
except (ModuleNotFoundError, ImportError) as e:
    from optimizer import Optimizer
    from init_cuda_error_fix import InitCudaErrorFix
    from seed import ManualSeed
    from cudnn import DisableCuDNN, DisableTensorCores
    from image import ImageFileInput
    from body import Body
    from metadata import Metadata
    from openpose import OpenPoseFile
    from silhouette_holes import SilhouetteHoles
    from klothed_v4_1 import KlothedBodyV4_1

log = logging.getLogger(__name__)

__all__ = ['KBodyServer']


#TODO: configure:
    #   1. shopper/product inner/outer edt
    #   2. holes weight
    #   3. virtual joint weights
    #   4. staged weights and priors for product/shopper
    #   5. throw on error
    #   6. 
class KBodyServer(BaseHandler):
    def __init__(self) -> None:
        super().__init__()

    #TODO: product/shopper differentiators
    def initialize(self, context):
        self._context = context
        self.manifest = context.manifest
        properties = context.system_properties                
        # self.map_location = "cuda" if torch.cuda.is_available() and properties.get("gpu_id") is not None else "cpu"
        self.device = torch.device('cpu')
        if torch.cuda.is_available() and not 'FORCE_CPU' in os.environ:
            gpu_id = properties.get("gpu_id")
            if gpu_id is not None:
                self.device = torch.device(f"cuda:{gpu_id}")
            else:
                self.device = torch.device("cuda")        
        log.info(f"Model set to run on a [{self.device}] device")        
        
        model_dir = properties.get("model_dir")
        config_file = self.manifest['model']['serializedFile']
        config_file = os.path.join(model_dir, config_file)        
        log.info(f"Loading configuration file from {config_file}")
        self.config = OmegaConf.load(config_file)

        self.init_cuda_error_fix = InitCudaErrorFix()
        self.seed = ManualSeed(1337, True)
        self.disable_cudnn = DisableCuDNN()
        self.disable_tensor_cores = DisableTensorCores()
        self.optimizer = Optimizer(self.config)
        
        self.optimizer = self.optimizer.to(self.device)
        self.optimizer.eval()
        self.initialized = True
        log.info(f"Model ({type(self.optimizer.model)}) loaded successfully.")
        
        self.gradient_tolerance = self.optimizer.tol_g
        self.relative_tolerance = self.optimizer.tol_e

        edt_params = {
            'shopper': { 'scale_inner': 0.75, 'scale_outer': 2.0,},
            'product': { 'scale_inner': 0.5, 'scale_outer': 2.0,},
        }
        self.preproc = {
            'image': ImageFileInput(),
            'body': Body(),
            'metadata': Metadata(),
            'openpose': OpenPoseFile(load_face_contour=True),
            'silhouette_holes': SilhouetteHoles(
                kernel_size=3, iterations=1,
                **edt_params[self.config.context.type]
            ),
        }
        self.postproc = {
            'klothed_v4.1': KlothedBodyV4_1(
                height_regressors=os.path.join(
                    self.config.body.data_root, 'assets', 'height_regressors'
                )
            )
        }

    def preprocess(self, 
        data:   typing.Mapping[str, typing.Any],
    ) -> typing.Dict[str, torch.Tensor]:
        log.debug(f"Preprocessing input:\n{data}")
        tensors = { 'json': data }
        body = data[0].get('body') or data[0].get('raw')
        for k, p in self.preproc.items():
            tensors = toolz.merge(tensors, p(body, self.device))
        # log.debug(f"Tensors: {tensors.keys()}")
        return tensors

    def relative_check(self, 
        prev: torch.Tensor, 
        current: torch.Tensor
    ) -> float:
        relative_change = (prev - current) / max([prev.abs(), current.abs(), 1.0])
        return relative_change <= self.relative_tolerance

    def gradient_check(self, 
        param_groups: typing.Sequence[typing.Dict[str, torch.nn.Parameter]]
    ) -> bool:
        return all(
            p.grad.view(-1).max().abs().item() < self.gradient_tolerance 
            for p in toolz.concat((g['params'] for g in param_groups)) 
            if p.grad is not None
        )

    def is_any_param_nan(self, optimizer: torch.optim.Optimizer) -> bool:
        for pg in optimizer.param_groups:
                for p in pg['params']:
                    if not torch.all(torch.isfinite(p)):
                        return True
        return

    def inference(self, 
        data:       typing.Mapping[str, torch.Tensor],
    ):        
        self.last_loss = None
        self.optimizer.initialize_parameters()
        optimizers = self.optimizer.configure_optimizers()
        iters = list(toolz.mapcat(lambda o: o.iterations, toolz.unique(optimizers)))
        stages = list(toolz.mapcat(lambda o: o.name, toolz.unique(optimizers)))           
        with torch.no_grad():
            self.optimizer.assign_predictions_to_params(data)
        for i, (optim, iters, stage) in enumerate(zip(
            optimizers, iters, stages
        )):
            log.info(f"Optimizing stage: {stage} for {iters} iterations")
            for n, p in self.optimizer.named_parameters():
                p.requires_grad_(False)
            for pg in optim.param_groups:
                for p in pg['params']:
                    p.requires_grad_(True)
            def closure():
                self.optimizer.optimizer_zero_grad(
                    epoch=0, batch_idx=0,
                    optimizer=optim, optimizer_idx=i
                )
                td = self.optimizer.training_step(batch=data, 
                    batch_idx=0, optimizer_idx=i
                )
                self.loss = td['loss']
                self.losses = td['losses']
                self.loss.backward()
                self.optimizer.optimization_step += 1
                data['optimization_step'] = self.optimizer.optimization_step
                return self.loss
            for j in range(iters):
                optim.step(closure=closure)                            
                current_loss = self.loss                
                if (self.last_loss is not None and self.relative_check(
                    self.last_loss, current_loss
                )) or self.gradient_check(optim.param_groups)\
                    or not torch.isfinite(current_loss)\
                    or self.is_any_param_nan(optim):
                        log.warning(f"Optimization stage '{stage}' stopped at iteration {j}/{iters}.")
                        break
                self.last_loss = current_loss
            self.last_loss = None
            self.optimizer.optimization_step = 0        
        metrics = self.optimizer.validation(data)
        if hasattr(self._context, 'metrics'):
            for k, v in metrics.items(): # self.context is set in base handler's handle method            
                self._context.metrics.add_metric(name=k, value=float(v.detach().cpu().numpy()), unit='value')
        else:            
            for k, v in metrics.items():
                log.info(f"Metric [{k}] = {float(v.detach().cpu().numpy())}")
        return data 
       
    def postprocess(self,
        data: typing.Mapping[str, torch.Tensor]
    ) -> typing.Sequence[typing.Any]:
        outs = [] #TODO: corner case with no postproc crashes, fix it
        for k, p in self.postproc.items():
            res = p(data, data['json'])
            if len(outs) == 0:
                outs = res
            else:                
                for o, r in zip(outs, res):
                    o = toolz.merge(o, r)
        return outs