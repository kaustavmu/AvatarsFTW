import torch
import typing
import logging
import toolz

log = logging.getLogger(__name__)

__all__ = ["ZeroBetas"]


def get_submodule_pt_ge_110(
    module: torch.nn.Module,
    name: str
) -> torch.nn.Module:
    return module.get_submodule(name)

def get_child_pt_lt_110(
    module: torch.nn.Module,
    name: str
) -> torch.nn.Module:
    split = name.split('.')
    def _getattr(object: typing.Any, key: str):
        return getattr(object, key, None)
    return toolz.reduce(_getattr, split, module)

if isinstance(torch.__version__, str):
    v = torch.__version__.split('.')
else:
    v = torch.__version__

if (int(v[0]), int(v[1])) >= (1, 10):
    get_submodule = get_submodule_pt_ge_110
else:
    get_submodule = get_child_pt_lt_110

class ZeroBetas(typing.Callable[[torch.nn.Module], None]):
    def __init__(self):
        pass
    
    def __call__(self,
        module: torch.nn.Module
    ) -> None:                
        try:            
            m = get_submodule(module, 'preprocess.betas')
            if m is not None:
                log.info(f"Zeroing out beta parameters.")
                with torch.no_grad():
                    for p in m.parameters():
                        p.zero_()
        except:
            pass