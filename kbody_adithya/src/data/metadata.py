import cv2
import torch
import glob
import numpy as np
import typing
import logging

__all__ = ["Metadata"]

log = logging.getLogger(__name__)

def load_color_image(
    filename:       str,    
    output_space:   str='norm', # ['norm', 'ndc', 'pixel']
) -> torch.Tensor:
    img = torch.from_numpy(
        cv2.imread(filename).transpose(2, 0, 1)
    ).flip(dims=[0])
    
    if output_space == 'norm':
        img = img / 255.0
    return img

class Metadata(torch.utils.data.Dataset):
    def __init__(self,
        metadata_glob:          str,
        image_glob:             str,
        focal_length:           float=5000.0,
    ):
        self.metadata_files = glob.glob(metadata_glob)
        self.image_files = glob.glob(image_glob)
        self.focal_length = focal_length
        log.info(f"Loaded {len(self)} .npz files.")

    def __len__(self) -> int:
        return len(self.metadata_files)

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:        
        data = np.load(self.metadata_files[index], allow_pickle=True)['metadata'].item()
        img = load_color_image(self.image_files[index])
        bottom_pad = data.get('bottom_padding', 0)
        log.info(f"Using a principal point that is offseted by {bottom_pad} pixels.")
        fx, fy = self.focal_length, self.focal_length
        return { 
            'camera_intrinsics': torch.Tensor(
                [
                    [fx, 0.0, img.shape[-1] / 2.0],
                    [0.0, fy, (img.shape[-2] - bottom_pad) / 2.0],
                    [0.0, 0.0, 1.0],
                ]
            ).float(),
        }
        