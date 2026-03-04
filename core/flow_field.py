"""
Flow Field — Learnable displacement field on top of an identity grid.

The identity grid maps each pixel to itself. A displacement tensor is
added on top and optimized via backpropagation. `torch.grid_sample`
applies the combined flow to warp the source image.
"""

import torch
import torch.nn.functional as F


class FlowField:
    """
    Manages the identity grid + learnable displacement field.

    The flow is:  flow = identity_grid + displacement
    Warping is:   warped = grid_sample(source, flow)

    Args:
        height:    Image height in pixels.
        width:     Image width in pixels.
        device:    Torch device to allocate tensors on.
        max_disp:  Maximum displacement magnitude (clamped after each step).
        mode:      Sampling mode — 'bilinear' (smooth) or 'nearest' (strict).
    """

    def __init__(
        self,
        height: int,
        width: int,
        device: torch.device,
        max_disp: float = 1.0,
        mode: str = "bilinear",
    ):
        self.height = height
        self.width = width
        self.device = device
        self.max_disp = max_disp
        self.mode = mode

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

    @property
    def flow(self) -> torch.Tensor:
        """Current flow grid = identity + displacement."""
        return self.identity_grid + self.displacement

    def warp(self, source: torch.Tensor) -> torch.Tensor:
        """
        Warp the source image using the current flow field.

        Args:
            source: (1, C, H, W) source image tensor.

        Returns:
            (1, C, H, W) warped image tensor.
        """
        return F.grid_sample(
            source,
            self.flow,
            mode=self.mode,
            padding_mode="border",
            align_corners=True,
        )

    def clamp_displacement(self):
        """Clamp displacement to [-max_disp, max_disp] in-place."""
        with torch.no_grad():
            self.displacement.clamp_(-self.max_disp, self.max_disp)

    def reset(self):
        """Reset displacement to zero (identity mapping)."""
        with torch.no_grad():
            self.displacement.zero_()
        self.displacement.requires_grad_(True)

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
