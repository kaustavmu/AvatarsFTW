import torch
import functools
import typing

def dim_list(
    tensor:         torch.Tensor,
    start_index:    int=1,
) -> typing.List[int]:
    return list(range(start_index, len(tensor.shape)))

def expand_dims(
    src:            torch.Tensor,
    dst:            torch.Tensor,
    start_index:    int=1,
) -> torch.Tensor:
    r"""
        Expands the source tensor to match the spatial dimensions of the destination tensor.
        
        Arguments:
            src (torch.Tensor): A tensor of [B, K, X(Y)(Z)] dimensions
            dst (torch.Tensor): A tensor of [B, X(Y)(Z), (D), (H), W] dimensions
            start_index (int, optional): An optional start index denoting the start of the spatial dimensions
        
        Returns:
            A torch.Tensor of [B, K, X(Y)(Z), (1), (1), 1] dimensions. 
    """
    return functools.reduce(
        lambda s, _: s.unsqueeze(-1), 
        [*dst.shape[start_index:]],
        src
    )

def flatten_spatial_dims(
    tensor:         torch.Tensor,
    spatial_start_index:    int=2,
) -> torch.Tensor:
    dims = [*tensor.shape[:spatial_start_index]] + [-1]
    return tensor.view(*dims)

spatial_dim_list = functools.partial(dim_list, start_index=2)

__all__ = ['IoU']

class IoU(torch.nn.Module):
    def __init__(self, 
        reduce: bool=False, # False when used as a loss, True when used as a metric
    ):
        super(IoU, self).__init__()
        self.reduce = reduce

    def forward(self, 
        pred:   torch.Tensor,
        gt:     torch.Tensor,        
    ) -> torch.Tensor:
        dims = spatial_dim_list(pred)
        intersect = (pred * gt).sum(dims)
        union = (pred + gt - (pred * gt)).sum(dims) + 1e-6
        return (
            (intersect / union).sum() / intersect.numel() #NOTE: is correct for batch size = 1 only
         ) if self.reduce else intersect / union