"""
Device Manager — Auto-detects GPUs and assigns devices for dual-GPU split.

GPU 0: Feature extractor (VGG19)
GPU 1: Optimizer + warping (displacement field, grid_sample)
Falls back gracefully to single GPU or CPU.
"""

import torch


class DeviceManager:
    """
    Manages device assignment for the dual-GPU architecture.

    With 2+ GPUs:
        - feature_device = cuda:0  (VGG19 runs here)
        - optim_device   = cuda:1  (displacement field + warping)
    With 1 GPU:
        - Both on cuda:0
    With 0 GPUs:
        - Both on CPU
    """

    def __init__(self, force_cpu: bool = False):
        self.num_gpus = torch.cuda.device_count() if not force_cpu else 0

        if self.num_gpus >= 2:
            self.feature_device = torch.device("cuda:0")
            self.optim_device = torch.device("cuda:1")
        elif self.num_gpus == 1:
            self.feature_device = torch.device("cuda:0")
            self.optim_device = torch.device("cuda:0")
        else:
            self.feature_device = torch.device("cpu")
            self.optim_device = torch.device("cpu")

    @property
    def is_cuda(self) -> bool:
        return self.num_gpus > 0

    @property
    def is_dual_gpu(self) -> bool:
        return self.num_gpus >= 2

    def summary(self) -> str:
        if self.is_dual_gpu:
            return (
                f"Dual-GPU mode: Features on {self.feature_device}, "
                f"Optimizer on {self.optim_device}"
            )
        elif self.is_cuda:
            return f"Single-GPU mode: All on {self.feature_device}"
        else:
            return "CPU mode: No GPU detected"

    def __repr__(self) -> str:
        return (
            f"DeviceManager(feature={self.feature_device}, "
            f"optim={self.optim_device}, gpus={self.num_gpus})"
        )
