import torch

__all__ = ['FixNeck']

class FixNeck(torch.nn.Module):#TODO: inplace annotation to check for errors using different in/out keys
    def __init__(self) -> None:
        super(FixNeck, self).__init__()

    def forward(self, 
        joints:     torch.Tensor, #NOTE: openpose coco25 assumption
    ) -> torch.Tensor:
        # avg(2, 5) -> 1
        joints[:, 1, :] = 0.5 * (joints[:, 2, :] + joints[:, 5, :])
        return joints