"""
Flow Field — Learnable motion field on top of an identity grid.

Supports:
1) Backward sampling (`bilinear` / `nearest`) via `torch.grid_sample`
2) Physical advection (`physical`) via semi-Lagrangian backtracing
   with optional rigid motion (rotation + translation).
"""

import torch
import torch.nn.functional as F


class FlowField:
    """
    Manages the identity grid + learnable displacement field.

    The flow is:  flow = identity_grid + displacement (+ rigid motion in physical mode)
    Warping is:   warped = grid_sample(...) or semi-Lagrangian advection

    Args:
        height:    Image height in pixels.
        width:     Image width in pixels.
        device:    Torch device to allocate tensors on.
        max_disp:  Maximum displacement magnitude (clamped after each step).
        mode:      'physical' (semi-Lagrangian advection), 'bilinear', or 'nearest'.
    """

    def __init__(
        self,
        height: int,
        width: int,
        device: torch.device,
        max_disp: float = 0.5,
        mode: str = "physical",
    ):
        self.height = height
        self.width = width
        self.device = device
        self.max_disp = max_disp
        self.mode = mode
        self.use_physical_motion = mode == "physical"
        self.physical_steps = 6
        self.max_translation = 0.5
        self.max_rotation = torch.pi / 2
        self.smoothing_blend = 0.22

        # Build identity grid: (1, H, W, 2) with values in [-1, 1]
        y = torch.linspace(-1.0, 1.0, height, device=device)
        x = torch.linspace(-1.0, 1.0, width, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        self.identity_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        # shape: (1, H, W, 2) — last dim is (x, y) as grid_sample expects

        # Learnable displacement, initialized to zero (no movement)
        self.displacement = torch.zeros(
            1, height, width, 2, device=device, requires_grad=True
        )

        # Rigid-body motion parameters (used in physical mode)
        self.rotation = torch.zeros(
            1, device=device, requires_grad=self.use_physical_motion
        )
        self.translation = torch.zeros(
            1, 1, 1, 2, device=device, requires_grad=self.use_physical_motion
        )

    @property
    def flow(self) -> torch.Tensor:
        """Current flow grid in normalized coordinates."""
        if not self.use_physical_motion:
            return self.identity_grid + self.displacement
        return self._forward_motion_grid()

    def _forward_motion_grid(self) -> torch.Tensor:
        """Source-to-destination motion in normalized coordinates."""
        x = self.identity_grid[..., 0]
        y = self.identity_grid[..., 1]
        cos_t = torch.cos(self.rotation).view(1, 1, 1)
        sin_t = torch.sin(self.rotation).view(1, 1, 1)
        x_rot = cos_t * x - sin_t * y
        y_rot = sin_t * x + cos_t * y
        rigid_grid = torch.stack([x_rot, y_rot], dim=-1)
        return rigid_grid + self.translation + self.displacement

    def _inverse_rigid_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Convert destination grid coordinates to source coordinates
        under the inverse rigid motion.
        """
        cos_t = torch.cos(self.rotation).view(1, 1, 1)
        sin_t = torch.sin(self.rotation).view(1, 1, 1)
        shifted = grid - self.translation
        x = shifted[..., 0]
        y = shifted[..., 1]
        x_inv = cos_t * x + sin_t * y
        y_inv = -sin_t * x + cos_t * y
        return torch.stack([x_inv, y_inv], dim=-1)

    def _semi_lagrangian_warp(self, source: torch.Tensor) -> torch.Tensor:
        """
        Physically advect pixels using destination-to-source backtracing.
        """
        grid = self._inverse_rigid_grid(self.identity_grid)
        velocity_field = self.displacement.permute(0, 3, 1, 2)
        dt = 1.0 / float(self.physical_steps)

        for _ in range(self.physical_steps):
            velocity = F.grid_sample(
                velocity_field,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ).permute(0, 2, 3, 1)
            grid = grid - dt * velocity

        grid = grid.clamp(-1.0, 1.0)
        return F.grid_sample(
            source,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    def warp(self, source: torch.Tensor) -> torch.Tensor:
        """
        Warp the source image using the current flow field.

        Args:
            source: (1, C, H, W) source image tensor.

        Returns:
            (1, C, H, W) warped image tensor.
        """
        if self.use_physical_motion:
            return self._semi_lagrangian_warp(source)

        return F.grid_sample(
            source,
            self.flow,
            mode=self.mode,
            padding_mode="border",
            align_corners=True,
        )

    def clamp_displacement(self):
        """Clamp motion parameters in-place for optimization stability."""
        with torch.no_grad():
            self.displacement.clamp_(-self.max_disp, self.max_disp)
            if self.use_physical_motion:
                self.translation.clamp_(-self.max_translation, self.max_translation)
                self.rotation.clamp_(-self.max_rotation, self.max_rotation)

    def smooth_displacement(self):
        """Diffuse high-frequency velocity noise for coherent particle motion."""
        if not self.use_physical_motion or self.smoothing_blend <= 0.0:
            return
        with torch.no_grad():
            disp = self.displacement.permute(0, 3, 1, 2)
            smooth = F.avg_pool2d(disp, kernel_size=5, stride=1, padding=2)
            disp = (1.0 - self.smoothing_blend) * disp + self.smoothing_blend * smooth
            self.displacement.copy_(disp.permute(0, 2, 3, 1))

    def reset(self):
        """Reset displacement to zero (identity mapping)."""
        with torch.no_grad():
            self.displacement.zero_()
            self.translation.zero_()
            self.rotation.zero_()
        self.displacement.requires_grad_(True)
        self.translation.requires_grad_(self.use_physical_motion)
        self.rotation.requires_grad_(self.use_physical_motion)

    def get_trainable_tensors(self) -> list:
        """Return tensors that should be optimized."""
        tensors = [self.displacement]
        if self.use_physical_motion:
            tensors.extend([self.rotation, self.translation])
        return tensors

    def get_displacement_magnitude(self) -> torch.Tensor:
        """Return per-pixel displacement magnitude for visualization."""
        with torch.no_grad():
            return torch.norm(self.displacement[0], dim=-1)

    def resize(self, new_height: int, new_width: int):
        """
        Resize the flow field to a new resolution.
        Useful for multi-scale optimization.
        """
        if new_height == self.height and new_width == self.width:
            return

        # Rebuild identity grid at new resolution
        y = torch.linspace(-1.0, 1.0, new_height, device=self.device)
        x = torch.linspace(-1.0, 1.0, new_width, device=self.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        self.identity_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        # Interpolate existing displacement to new resolution
        # displacement shape: (1, H, W, 2) → permute to (1, 2, H, W) for interpolate
        disp_permuted = self.displacement.detach().permute(0, 3, 1, 2)
        disp_resized = F.interpolate(
            disp_permuted,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=True,
        )
        self.displacement = disp_resized.permute(0, 2, 3, 1).contiguous()
        self.displacement.requires_grad_(True)

        self.height = new_height
        self.width = new_width
