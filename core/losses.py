"""
Loss Functions — Sinkhorn/SWD, Perceptual (VGG), and Total Variation.

Three complementary losses that drive the displacement field to rearrange
source pixels into the target shape while preserving color distributions
and ensuring smooth, fluid-like motion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import geomloss for GPU-accelerated Sinkhorn
try:
    from geomloss import SamplesLoss

    HAS_GEOMLOSS = True
except ImportError:
    HAS_GEOMLOSS = False


# ---------------------------------------------------------------------------
# Sliced Wasserstein Distance (lightweight OT fallback)
# ---------------------------------------------------------------------------

def sliced_wasserstein_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    n_projections: int = 128,
) -> torch.Tensor:
    """
    Sliced Wasserstein Distance between two point clouds.

    Projects onto random 1D directions, sorts, and computes L2.
    O(N log N) per projection — much cheaper than full Sinkhorn.

    Args:
        x: (N, D) source point cloud (flattened pixel colors).
        y: (M, D) target point cloud (flattened pixel colors).
        n_projections: Number of random projections to average over.

    Returns:
        Scalar SWD loss.
    """
    d = x.shape[1]
    device = x.device

    # Random unit directions on the D-sphere
    directions = torch.randn(n_projections, d, device=device)
    directions = F.normalize(directions, dim=1)

    # Project both point clouds: (n_proj, N) and (n_proj, M)
    proj_x = directions @ x.t()  # (n_proj, N)
    proj_y = directions @ y.t()  # (n_proj, M)

    # Sort along the point dimension
    proj_x_sorted, _ = torch.sort(proj_x, dim=1)
    proj_y_sorted, _ = torch.sort(proj_y, dim=1)

    # If point clouds have different sizes, interpolate to match
    if proj_x_sorted.shape[1] != proj_y_sorted.shape[1]:
        n = max(proj_x_sorted.shape[1], proj_y_sorted.shape[1])
        proj_x_sorted = F.interpolate(
            proj_x_sorted.unsqueeze(1), size=n, mode="linear", align_corners=True
        ).squeeze(1)
        proj_y_sorted = F.interpolate(
            proj_y_sorted.unsqueeze(1), size=n, mode="linear", align_corners=True
        ).squeeze(1)

    # Mean L2 distance between sorted projections
    return torch.mean((proj_x_sorted - proj_y_sorted) ** 2)


# ---------------------------------------------------------------------------
# Sinkhorn Loss (uses geomloss if available, else SWD)
# ---------------------------------------------------------------------------

class SinkhornLoss(nn.Module):
    """
    Optimal transport loss for color distribution matching.

    Uses geomloss.SamplesLoss (entropy-regularized Sinkhorn) when available,
    falls back to Sliced Wasserstein Distance otherwise.

    Args:
        blur:           Sinkhorn blur/epsilon parameter.
        n_projections:  Number of SWD projections (fallback mode).
        max_points:     Subsample to this many pixels for memory efficiency.
    """

    def __init__(
        self,
        blur: float = 0.05,
        n_projections: int = 128,
        max_points: int = 16384,
    ):
        super().__init__()
        self.blur = blur
        self.n_projections = n_projections
        self.max_points = max_points
        self.use_geomloss = HAS_GEOMLOSS

        if self.use_geomloss:
            self.loss_fn = SamplesLoss(
                loss="sinkhorn",
                p=2,
                blur=blur,
                scaling=0.9,
                backend="tensorized",
            )

    def _subsample(self, pixels: torch.Tensor) -> torch.Tensor:
        """Randomly subsample pixels if exceeding max_points."""
        n = pixels.shape[0]
        if n <= self.max_points:
            return pixels
        indices = torch.randperm(n, device=pixels.device)[: self.max_points]
        return pixels[indices]

    def forward(
        self, warped: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute OT loss between warped and target images.

        Args:
            warped: (1, 3, H, W) warped source image.
            target: (1, 3, H, W) target image.

        Returns:
            Scalar loss value.
        """
        # Flatten to (N, 3) point clouds
        w_pixels = warped[0].permute(1, 2, 0).reshape(-1, 3)  # (H*W, 3)
        t_pixels = target[0].permute(1, 2, 0).reshape(-1, 3)

        # Subsample for memory efficiency
        w_pixels = self._subsample(w_pixels)
        t_pixels = self._subsample(t_pixels)

        if self.use_geomloss:
            return self.loss_fn(w_pixels, t_pixels)
        else:
            return sliced_wasserstein_distance(
                w_pixels, t_pixels, self.n_projections
            )


# ---------------------------------------------------------------------------
# Perceptual Loss (VGG feature matching)
# ---------------------------------------------------------------------------

class PerceptualLoss(nn.Module):
    """
    Multi-layer VGG feature matching loss.

    Computes MSE between features extracted from the warped image and the
    pre-computed target features at selected VGG layers.

    Args:
        layer_weights: Optional dict of {layer_name: weight}. If None,
                       all layers are weighted equally.
    """

    def __init__(self, layer_weights: dict = None):
        super().__init__()
        self.layer_weights = layer_weights or {}

    def forward(
        self,
        warped_features: dict,
        target_features: dict,
    ) -> torch.Tensor:
        """
        Compute perceptual loss between feature dicts.

        Args:
            warped_features: {layer_name: tensor} from VGG on warped image.
            target_features: {layer_name: tensor} from VGG on target image.

        Returns:
            Scalar perceptual loss.
        """
        loss = torch.tensor(0.0, device=next(iter(warped_features.values())).device)

        for name in warped_features:
            if name not in target_features:
                continue
            w = self.layer_weights.get(name, 1.0)
            # Move target features to same device if needed (dual-GPU)
            tf = target_features[name].to(warped_features[name].device)
            loss = loss + w * F.mse_loss(warped_features[name], tf)

        return loss


# ---------------------------------------------------------------------------
# Total Variation Loss on the displacement field
# ---------------------------------------------------------------------------

def total_variation_loss(displacement: torch.Tensor) -> torch.Tensor:
    """
    Total Variation penalty on the displacement field.

    Encourages smooth, fluid-like pixel motion by penalizing
    large differences between neighboring displacement vectors.

    Args:
        displacement: (1, H, W, 2) displacement tensor.

    Returns:
        Scalar TV loss.
    """
    # Spatial gradients in y and x directions
    diff_y = displacement[:, 1:, :, :] - displacement[:, :-1, :, :]
    diff_x = displacement[:, :, 1:, :] - displacement[:, :, :-1, :]
    return torch.mean(diff_y ** 2) + torch.mean(diff_x ** 2)


# ---------------------------------------------------------------------------
# Combined Loss
# ---------------------------------------------------------------------------

class PixelShiftLoss(nn.Module):
    """
    Combined loss: Sinkhorn + Perceptual + Total Variation.

    Args:
        device:          Device for computation.
        w_sinkhorn:      Weight for OT / color distribution loss.
        w_perceptual:    Weight for VGG perceptual loss.
        w_tv:            Weight for total variation on displacement.
        sinkhorn_blur:   Blur parameter for Sinkhorn.
        sinkhorn_points: Max subsample points for Sinkhorn.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        w_sinkhorn: float = 1.0,
        w_perceptual: float = 1.0,
        w_tv: float = 0.1,
        sinkhorn_blur: float = 0.05,
        sinkhorn_points: int = 16384,
    ):
        super().__init__()
        self.w_sinkhorn = w_sinkhorn
        self.w_perceptual = w_perceptual
        self.w_tv = w_tv
        self.device = device

        self.sinkhorn_loss = SinkhornLoss(
            blur=sinkhorn_blur, max_points=sinkhorn_points
        )
        self.perceptual_loss = PerceptualLoss()

    def forward(
        self,
        warped: torch.Tensor,
        target: torch.Tensor,
        displacement: torch.Tensor,
        warped_features: dict = None,
        target_features: dict = None,
    ) -> dict:
        """
        Compute all loss components and return as a dictionary.

        Args:
            warped:           (1, 3, H, W) warped source image.
            target:           (1, 3, H, W) target image.
            displacement:     (1, H, W, 2) displacement field.
            warped_features:  VGG features of warped image (optional).
            target_features:  VGG features of target image (optional).

        Returns:
            Dict with keys: 'sinkhorn', 'perceptual', 'tv', 'total'.
        """
        losses = {}

        # --- Sinkhorn / SWD loss ---
        loss_sink = self.sinkhorn_loss(warped, target)
        losses["sinkhorn"] = loss_sink

        # --- Perceptual loss ---
        if warped_features is not None and target_features is not None:
            loss_perc = self.perceptual_loss(warped_features, target_features)
        else:
            loss_perc = torch.tensor(0.0, device=self.device)
        losses["perceptual"] = loss_perc

        # --- Total Variation loss ---
        loss_tv = total_variation_loss(displacement)
        losses["tv"] = loss_tv

        # --- Combined ---
        total = (
            self.w_sinkhorn * loss_sink
            + self.w_perceptual * loss_perc
            + self.w_tv * loss_tv
        )
        losses["total"] = total

        return losses
