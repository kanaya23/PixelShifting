"""
Optimizer Engine — Pixel-drift reconstruction via physics simulation or
gradient-based optimization.

Physical mode:
    1. Canny edge detection on the target image.
    2. Colour-based pixel assignment (edges first, then luminance sort).
    3. Spring-damper particle simulation moving source pixels to targets.

Gradient mode (bilinear / nearest):
    Classical displacement-field optimisation with SWD + VGG + TV losses.
"""

import threading
from typing import Callable, Optional

import torch
import torch.optim as optim

from .device_manager import DeviceManager
from .flow_field import FlowField
from .feature_extractor import VGGFeatureExtractor
from .losses import PixelShiftLoss
from .pixel_engine import PixelAssigner, PhysicsSimulator


class OptimizerEngine:
    """
    Orchestrates pixel-drift reconstruction.

    In *physical* mode the engine bypasses gradient optimisation entirely and
    runs a deterministic Canny-guided assignment followed by a spring-damper
    particle simulation.  In *bilinear*/*nearest* mode it falls back to the
    classical gradient loop.

    Args:
        source / target:  (1, 3, H, W) image tensors.
        device_manager:   DeviceManager instance.
        lr:               Learning rate (gradient mode only).
        iterations:       Number of physics timesteps or gradient steps.
        sampling_mode:    'physical', 'bilinear', or 'nearest'.
        update_interval:  Emit progress every N steps.
        on_progress:      Callback(step, total, warped_tensor, loss_dict).
        on_finished:      Callback(final_warped_tensor).
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
        sampling_mode: str = "physical",
        max_displacement: float = 0.5,
        update_interval: int = 10,
        dist_mode: str = "swd",
        on_progress: Optional[Callable] = None,
        on_finished: Optional[Callable] = None,
    ):
        self.device_manager = device_manager
        self.iterations = iterations
        self.update_interval = update_interval
        self.on_progress = on_progress
        self.on_finished = on_finished
        self.sampling_mode = sampling_mode

        dev = device_manager.optim_device
        self.source = source.to(dev)
        self.target = target.to(dev)

        # ── Physical mode setup ──
        self._physics_sim: Optional[PhysicsSimulator] = None
        if sampling_mode == "physical":
            assigner = PixelAssigner()
            assignment = assigner.assign(self.source, self.target)
            self._physics_sim = PhysicsSimulator(
                source=self.source,
                assignment=assignment,
                device=dev,
            )

        # ── Gradient mode setup ──
        self.flow_field = None
        self.vgg = None
        self.target_features = None
        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None

        if sampling_mode != "physical":
            _, _, h, w = source.shape
            self.flow_field = FlowField(
                height=h, width=w, device=dev,
                max_disp=max_displacement, mode=sampling_mode,
            )
            self.vgg = VGGFeatureExtractor(device=device_manager.feature_device)
            with torch.no_grad():
                feat_raw = self.vgg(self.target.to(device_manager.feature_device))
            self.target_features = {
                k: v.detach().to(dev) for k, v in feat_raw.items()
            }
            self.loss_fn = PixelShiftLoss(
                device=dev, w_sinkhorn=w_sinkhorn,
                w_perceptual=w_perceptual, w_tv=w_tv, dist_mode=dist_mode,
            )
            self.optimizer = optim.Adam([self.flow_field.displacement], lr=lr)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=iterations,
                eta_min=min(lr * 0.01, 1e-4),
            )

        # Control flags
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._stop_flag = False
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    def pause(self):
        self._pause_event.clear()

    def resume(self):
        self._pause_event.set()

    def stop(self):
        self._stop_flag = True
        self._pause_event.set()

    # ── Main entry point ──

    def run(self):
        self._is_running = True
        self._stop_flag = False
        try:
            if self.sampling_mode == "physical":
                self._run_physics()
            else:
                self._run_gradient()
        finally:
            self._is_running = False

    # ── Physics simulation loop ──

    def _run_physics(self):
        sim = self._physics_sim
        try:
            for step in range(1, self.iterations + 1):
                self._pause_event.wait()
                if self._stop_flag:
                    break

                sim.step()

                if self.on_progress and step % self.update_interval == 0:
                    preview = sim.render().detach().cpu()
                    conv = sim.convergence()
                    loss_dict = {
                        "convergence": conv,
                        "total": 1.0 - conv,
                    }
                    self.on_progress(step, self.iterations, preview, loss_dict)
        finally:
            if self.on_finished:
                final = sim.render().detach().cpu()
                self.on_finished(final)

    # ── Gradient optimisation loop (bilinear / nearest) ──

    def _run_gradient(self):
        dev = self.device_manager.optim_device
        feat_dev = self.device_manager.feature_device
        try:
            for step in range(1, self.iterations + 1):
                self._pause_event.wait()
                if self._stop_flag:
                    break

                self.optimizer.zero_grad()
                warped = self.flow_field.warp(self.source)

                warped_feat = {
                    k: v.to(dev)
                    for k, v in self.vgg(warped.to(feat_dev)).items()
                }

                progress = (step - 1) / max(self.iterations - 1, 1)
                losses = self.loss_fn(
                    warped=warped, target=self.target,
                    displacement=self.flow_field.displacement,
                    warped_features=warped_feat,
                    target_features=self.target_features,
                    tv_weight_scale=progress,
                )

                losses["total"].backward()
                self.optimizer.step()
                self.scheduler.step()
                self.flow_field.clamp_displacement()

                if self.on_progress and step % self.update_interval == 0:
                    with torch.no_grad():
                        preview = warped.detach().cpu()
                        loss_dict = {k: v.item() for k, v in losses.items()}
                    self.on_progress(step, self.iterations, preview, loss_dict)
        finally:
            if self.on_finished:
                with torch.no_grad():
                    final = self.flow_field.warp(self.source).detach().cpu()
                self.on_finished(final)

    def get_current_result(self) -> torch.Tensor:
        if self._physics_sim is not None:
            return self._physics_sim.render().detach().cpu()
        with torch.no_grad():
            return self.flow_field.warp(self.source).detach().cpu()
