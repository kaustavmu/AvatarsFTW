import torch

__all__ = ['JabHips']

class JabHips(torch.nn.Module):
    def __init__(self,
        blend:              float=0.7,
        root_index:         int=0,
        left_hip_index:     int=1,
        right_hip_index:    int=4,
        version:            str=1.1,
    ) -> None:
        super(JabHips, self).__init__()
        self.root_index = root_index
        self.left_hip_index = left_hip_index
        self.right_hip_index = right_hip_index
        self.blend = blend

    def forward(self, 
        joints:     torch.Tensor,
    ) -> torch.Tensor:
        # joints[:, self.left_hip_index, :] = \
        #     self.blend * (joints[:, self.left_hip_index, :] + \
        #     (1.0 - self.blend) * joints[:, self.root_index, :])
        # joints[:, self.right_hip_index, :] = \
        #     self.blend * (joints[:, self.right_hip_index, :] + \
        #     (1.0 - self.blend) * joints[:, self.root_index, :])
        joints[:, self.left_hip_index, :] = \
            joints[:, self.left_hip_index, :] + self.blend * (
            joints[:, self.root_index, :] - joints[:, self.left_hip_index, :]
        )
        joints[:, self.right_hip_index, :] = \
            joints[:, self.right_hip_index, :] + self.blend * (
            joints[:, self.root_index, :] - joints[:, self.right_hip_index, :]
        )
        return joints