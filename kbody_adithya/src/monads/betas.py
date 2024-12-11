import torch

__all__ = ['Betas']

class Betas(torch.nn.Module):
    def __init__(self,
        count:      int=300,
        batch_size: int=1,
    ):
        super().__init__()
        for i in range(count):
            self.register_parameter(f"{i}", torch.nn.Parameter(
                torch.zeros(batch_size, 1)
            ))
    def forward(self, void: torch.Tensor) -> torch.nn.parameter.Parameter:
        return torch.cat(list(self.parameters()), dim=-1)
