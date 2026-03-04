"""
Optimizer Engine — Main optimization loop for pixel-drift reconstruction.

Runs in a background thread, optimizing the displacement field by
warping the source image and comparing it to the target via combined
Sinkhorn + Perceptual + TV losses. Emits progress callbacks for GUI.
"""

import time
import threading
from typing import Callable, Optional

import torch
import torch.optim as optim

from .device_manager import DeviceManager
from .flow_field import FlowField
from .feature_extractor import VGGFeatureExtractor
from .losses import PixelShiftLoss


class OptimizerEngine:
    """
    Orchestrates the pixel-drift optimization loop.

    Args:
        source:          (1, 3, H, W) source image tensor.
        target:          (1, 3, H, W) target image tensor.
        device_manager:  DeviceManager instance.
        lr:              Learning rate for Adam.
        iterations:      Total number of optimization steps.
        w_sinkhorn:      Weight for Sinkhorn/SWD loss.
        w_perceptual:    Weight for perceptual loss.
        w_tv:            Weight for total variation loss.
        sampling_mode:   'bilinear' or 'nearest'.
        max_displacement: Maximum displacement magnitude.
        update_interval: Emit progress every N steps.
        on_progress:     Callback(step, total, warped_tensor, loss_dict).
        on_finished:     Callback(final_warped_tensor).
    """

    def __init__(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        device_manager: DeviceManager,
        lr: float = 0.01,
        iterations: int = 1000,
        w_sinkhorn: float = 1.0,
        w_perceptual: float = 1.0,
        w_tv: float = 0.1,
        sampling_mode: str = "bilinear",
        max_displacement: float = 1.0,
        update_interval: int = 10,
        on_progress: Optional[Callable] = None,
        on_finished: Optional[Callable] = None,
    ):
        self.device_manager = device_manager
        self.iterations = iterations
        self.update_interval = update_interval
        self.on_progress = on_progress
        self.on_finished = on_finished

        # Move images to optimizer device
        dev = device_manager.optim_device
        self.source = source.to(dev)
        self.target = target.to(dev)

        _, _, h, w = source.shape

        # Flow field (displacement lives on optimizer device)
        self.flow_field = FlowField(
            height=h,
            width=w,
            device=dev,
            max_disp=max_displacement,
            mode=sampling_mode,
        )

        # VGG feature extractor (on feature device)
        self.vgg = VGGFeatureExtractor(device=device_manager.feature_device)

        # Pre-compute target features (once, read-only)
        self.target_features = self.vgg(self.target.to(device_manager.feature_device))

        # Loss function
        self.loss_fn = PixelShiftLoss(
            device=dev,
            w_sinkhorn=w_sinkhorn,
            w_perceptual=w_perceptual,
            w_tv=w_tv,
        )

        # Optimizer — directly optimizes the displacement tensor
        self.optimizer = optim.Adam([self.flow_field.displacement], lr=lr)

        # Learning rate scheduler (cosine annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=iterations, eta_min=lr * 0.01
        )

        # Control flags
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused by default
        self._stop_flag = False
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    def pause(self):
        """Pause the optimization loop."""
        self._pause_event.clear()

    def resume(self):
        """Resume the optimization loop."""
        self._pause_event.set()

    def stop(self):
        """Stop the optimization loop."""
        self._stop_flag = True
        self._pause_event.set()  # Unblock if paused

    def run(self):
        """
        Execute the optimization loop.

        This method should be called from a background thread.
        It optimizes the displacement field by minimizing the combined loss.
        """
        self._is_running = True
        self._stop_flag = False
        dev = self.device_manager.optim_device
        feat_dev = self.device_manager.feature_device

        try:
            for step in range(1, self.iterations + 1):
                # Check pause
                self._pause_event.wait()

                # Check stop
                if self._stop_flag:
                    break

                self.optimizer.zero_grad()

                # 1. Warp the source image
                warped = self.flow_field.warp(self.source)

                # 2. Extract features from warped image
                warped_features = self.vgg(warped.to(feat_dev))

                # Move warped features to optimizer device for loss
                warped_features_dev = {
                    k: v.to(dev) for k, v in warped_features.items()
                }
                target_features_dev = {
                    k: v.to(dev) for k, v in self.target_features.items()
                }

                # 3. Compute losses
                losses = self.loss_fn(
                    warped=warped,
                    target=self.target,
                    displacement=self.flow_field.displacement,
                    warped_features=warped_features_dev,
                    target_features=target_features_dev,
                )

                # 4. Backprop into displacement
                losses["total"].backward()
                self.optimizer.step()
                self.scheduler.step()

                # 5. Clamp displacement
                self.flow_field.clamp_displacement()

                # 6. Emit progress
                if self.on_progress and step % self.update_interval == 0:
                    with torch.no_grad():
                        preview = warped.detach().cpu()
                        loss_dict = {
                            k: v.item() for k, v in losses.items()
                        }
                    self.on_progress(step, self.iterations, preview, loss_dict)

        finally:
            self._is_running = False

            # Emit final result
            if self.on_finished:
                with torch.no_grad():
                    final_warped = self.flow_field.warp(self.source).detach().cpu()
                self.on_finished(final_warped)

    def get_current_result(self) -> torch.Tensor:
        """Get the current warped result (thread-safe read)."""
        with torch.no_grad():
            return self.flow_field.warp(self.source).detach().cpu()
