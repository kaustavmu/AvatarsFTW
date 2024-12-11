import torch
import typing
import toolz

__all__ = ['BetaRangeSelector']

def _get_submodule_pt_ge_110(
    module: torch.nn.Module,
    name: str
) -> torch.nn.Module:
    return module.get_submodule(name)    

def _get_child_pt_lt_110(
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
    get_submodule = _get_submodule_pt_ge_110
else:
    get_submodule = _get_child_pt_lt_110

class BetaRangeSelector(typing.Callable[[torch.nn.Module], typing.List[torch.Tensor]]):
    def __init__(self,
        key:             str,        
        start:           int=1,
        stop:            int=10,        
    ):
        self.key = key
        self.start = start
        self.stop = stop

    def __call__(self, module: torch.nn.Module) -> typing.List[torch.Tensor]:
        m = get_submodule(module, self.key)
        # return list(m.parameters())[self.start:self.stop]
        return { 'params': list(m.parameters())[self.start:self.stop] }