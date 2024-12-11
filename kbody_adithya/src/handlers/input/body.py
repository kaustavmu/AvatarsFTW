from collections.abc import Callable

import numpy as np
import typing
import torch
import pickle
import logging
import toolz

log = logging.getLogger(__name__)

__all__ = ['Body']

def load_pkl_file(
    filename:   str,
) -> typing.Dict[str, torch.Tensor]:
    data = { }
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

class Body(Callable):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, 
        data:   typing.Mapping[str, typing.Any],
        device: torch.device,
    ) -> typing.Dict[str, torch.Tensor]:
        filename = data.get('body')
        log.info(f"Loading SMPLX parameters file [key: body] @ {filename}")
        data = load_pkl_file(filename)
        with open(filename, 'rb') as body_file:
            body_data = pickle.load(body_file)
        #TODO: remove new axis after batchify preprocess
        return toolz.valmap(lambda t: (
                torch.from_numpy(t) if isinstance(t, np.ndarray) else t
            )[np.newaxis, ...].to(device), 
            body_data
        )