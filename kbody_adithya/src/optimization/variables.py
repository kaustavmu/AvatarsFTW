import torch

__all__ = ['Pose', 'Shape']

class Pose(torch.nn.Module):
    def __init__(self,
    ):
        super(Pose, self).__init__()        
        self.register_parameter("expression_t", 
            torch.nn.Parameter(torch.zeros(10).unsqueeze(0))
        )
        self.register_parameter("global_orient_t", 
            torch.nn.Parameter(torch.zeros(3).unsqueeze(0))
        )
        self.register_parameter("translation_t", 
            torch.nn.Parameter(torch.zeros(3).unsqueeze(0))
        )
        self.register_parameter("pose_t", 
            torch.nn.Parameter(torch.zeros(32).unsqueeze(0))
        )
        self.register_parameter("lhand_t", 
            torch.nn.Parameter(torch.zeros(12).unsqueeze(0))
        )
        self.register_parameter("rhand_t", 
            torch.nn.Parameter(torch.zeros(12).unsqueeze(0))
        )
        self.register_parameter("jaw_t", 
            torch.nn.Parameter(torch.zeros(3).unsqueeze(0))
        )

    def forward(self) -> torch.nn.parameter.Parameter:
        return dict(self.named_parameters())

class Shape(torch.nn.Module):
    def __init__(self,
        count:      int=300,
    ):
        super().__init__()
        for i in range(count):
            self.register_parameter(f"{i}", torch.nn.Parameter(
                torch.zeros(1, 1)
            ))
    def forward(self) -> torch.nn.parameter.Parameter:
        return torch.cat(list(self.parameters()), dim=-1)
