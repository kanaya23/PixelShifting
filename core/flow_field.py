"""
Flow Field — Learnable motion field on top of an identity grid.

Supports:
1) Backward sampling (`bilinear` / `nearest`) via `torch.grid_sample`
2) Physical forward advection (`physical`) using mass-preserving splatting
   with optional rigid motion (rotation + translation).
"""

import torch
import torch.nn.functional as F


class FlowField:
    """
    Manages the identity grid + learnable displacement field.

    The flow is:  flow = identity_grid + displacement (+ rigid motion in physical mode)
    Warping is:   warped = grid_sample(...) or differentiable forward splat

    Args:
        height:    Image height in pixels.
        width:     Image width in pixels.
        device:    Torch device to allocate tensors on.
        max_disp:  Maximum displacement magnitude (clamped after each step).
        mode:      'physical' (mass-preserving advection), 'bilinear', or 'nearest'.
    """

    def __init__(
        self,
        height: int,
        width: int,
        device: torch.device,
        max_disp: float = 0.35,
        mode: str = "physical",
    ):
        self.height = height
        self.width = width
        self.device = device
        self.max_disp = max_disp
        self.mode = mode
        self.use_physical_motion = mode == "physical"

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

    def _forward_splat_warp(self, source: torch.Tensor) -> torch.Tensor:
        """
        Physically advect source pixels with differentiable bilinear splatting.

        Each source pixel contributes its color mass to neighboring destination
        pixels based on continuous motion coordinates.
        """
        _, channels, h, w = source.shape
        flow = self._forward_motion_grid()

        x = (flow[..., 0] + 1.0) * 0.5 * (w - 1)
        y = (flow[..., 1] + 1.0) * 0.5 * (h - 1)

        x0 = torch.floor(x)
        y0 = torch.floor(y)
        x1 = x0 + 1.0
        y1 = y0 + 1.0

        wx1 = x - x0
        wy1 = y - y0
        wx0 = 1.0 - wx1
        wy0 = 1.0 - wy1

        src_flat = source[0].permute(1, 2, 0).reshape(-1, channels)
        accum = torch.zeros(h * w, channels, device=source.device, dtype=source.dtype)
        mass = torch.zeros(h * w, 1, device=source.device, dtype=source.dtype)

        def splat(x_idx: torch.Tensor, y_idx: torch.Tensor, weight: torch.Tensor):
            valid = (
                (x_idx >= 0.0)
                & (x_idx <= (w - 1))
                & (y_idx >= 0.0)
                & (y_idx <= (h - 1))
            )
            x_safe = x_idx.clamp(0.0, float(w - 1)).long()
            y_safe = y_idx.clamp(0.0, float(h - 1)).long()
            linear_idx = (y_safe * w + x_safe).reshape(-1)
            w_flat = (weight * valid.to(weight.dtype)).reshape(-1, 1)
            accum.scatter_add_(
                0,
                linear_idx.unsqueeze(1).expand(-1, channels),
                src_flat * w_flat,
            )
            mass.scatter_add_(0, linear_idx.unsqueeze(1), w_flat)

        splat(x0, y0, wx0 * wy0)
        splat(x1, y0, wx1 * wy0)
        splat(x0, y1, wx0 * wy1)
        splat(x1, y1, wx1 * wy1)

        warped_flat = accum / mass.clamp_min(1e-6)
        return warped_flat.view(h, w, channels).permute(2, 0, 1).unsqueeze(0)

    def warp(self, source: torch.Tensor) -> torch.Tensor:
        """
        Warp the source image using the current flow field.

        Args:
            source: (1, C, H, W) source image tensor.

        Returns:
            (1, C, H, W) warped image tensor.
        """
        if self.use_physical_motion:
            return self._forward_splat_warp(source)

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
                self.translation.clamp_(-1.0, 1.0)
                self.rotation.clamp_(-torch.pi, torch.pi)

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
