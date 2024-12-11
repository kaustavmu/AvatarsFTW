import torch
import logging

log = logging.getLogger(__name__)

__all__ = ['InitCudaErrorFix']

class InitCudaErrorFix(object):
    def __init__(self, 
        device:     str="cuda:0",
        iterations: int=100,
    ):
        Rt = torch.eye(4).float().to(device).unsqueeze(0)
        Rt[:, 3, :3] = torch.randn(3).to(device)        
        for _ in range(iterations):            
            try:
                dummy, e = torch.linalg.inv_ex(Rt)
                if e:
                    log.info(f"CUDA matrix inverse error ({e}).")
                    break
                dummy = torch.inverse(Rt)
                dummy = torch.linalg.inv(Rt)
            except e:
                log.info(f"CUDA matrix inverse error ({e}).")
                break
        log.info(f"Performed {iterations} iterations for CUDA matrix ops, no error.")