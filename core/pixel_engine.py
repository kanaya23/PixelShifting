"""
Pixel Engine — Canny-guided pixel assignment + GPU physics simulation.

1) CannyEdgeDetector  — Pure-PyTorch Canny edge detection on the target.
2) PixelAssigner      — Assigns each source pixel to a target position,
                        prioritising edge pixels for structural fidelity.
3) PhysicsSimulator   — Spring-damper particle system that moves source
                        pixels toward their assigned targets on GPU.
"""

import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Canny Edge Detector (pure PyTorch, no OpenCV dependency)
# ---------------------------------------------------------------------------

class CannyEdgeDetector:
    """
    GPU-friendly Canny edge detection.

    Pipeline: Gaussian blur → Sobel gradients → non-max suppression
              → double threshold → hysteresis.
    """

    def __init__(
        self,
        sigma: float = 1.4,
        low_threshold: float = 0.05,
        high_threshold: float = 0.15,
    ):
        self.sigma = sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    # -- kernels --------------------------------------------------------

    @staticmethod
    def _gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
        size = int(2 * math.ceil(2 * sigma) + 1)
        x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
        g = torch.exp(-x ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        return g.view(1, 1, 1, -1) * g.view(1, 1, -1, 1)

    @staticmethod
    def _sobel_kernels(device: torch.device):
        kx = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32, device=device,
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32, device=device,
        ).view(1, 1, 3, 3)
        return kx, ky

    # -- forward ---------------------------------------------------------

    @torch.no_grad()
    def detect(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run Canny on a (1, 3, H, W) or (1, 1, H, W) image tensor in [0, 1].

        Returns:
            (H, W) bool tensor — True at edge pixels.
        """
        device = image.device

        # Convert to grayscale if needed
        if image.shape[1] == 3:
            gray = (
                0.2989 * image[:, 0:1]
                + 0.5870 * image[:, 1:2]
                + 0.1140 * image[:, 2:3]
            )
        else:
            gray = image[:, 0:1]

        # 1. Gaussian blur
        gauss = self._gaussian_kernel(self.sigma, device)
        pad = gauss.shape[-1] // 2
        blurred = F.conv2d(F.pad(gray, [pad] * 4, mode="reflect"), gauss)

        # 2. Sobel gradients
        kx, ky = self._sobel_kernels(device)
        gx = F.conv2d(F.pad(blurred, [1] * 4, mode="reflect"), kx)
        gy = F.conv2d(F.pad(blurred, [1] * 4, mode="reflect"), ky)
        magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        angle = torch.atan2(gy, gx)

        # Normalise magnitude to [0, 1]
        mag_max = magnitude.max()
        if mag_max > 0:
            magnitude = magnitude / mag_max

        # 3. Non-maximum suppression (quantised to 4 directions)
        mag = magnitude[0, 0]
        ang = angle[0, 0]
        h, w = mag.shape
        nms = torch.zeros_like(mag)

        # Quantise angle to 0, 45, 90, 135
        ang_deg = (ang * 180.0 / math.pi) % 180.0
        padmag = F.pad(mag.unsqueeze(0).unsqueeze(0), [1] * 4, mode="constant")[0, 0]

        # direction bins
        d0 = ((ang_deg < 22.5) | (ang_deg >= 157.5))
        d45 = ((ang_deg >= 22.5) & (ang_deg < 67.5))
        d90 = ((ang_deg >= 67.5) & (ang_deg < 112.5))
        d135 = ((ang_deg >= 112.5) & (ang_deg < 157.5))

        # padmag is (h+2, w+2); original pixels at [1:-1, 1:-1]
        c = padmag[1:-1, 1:-1]

        # Compare along each direction pair and write surviving magnitudes
        survive_0 = (c >= padmag[1:-1, 2:]) & (c >= padmag[1:-1, :-2])
        survive_45 = (c >= padmag[:-2, 2:]) & (c >= padmag[2:, :-2])
        survive_90 = (c >= padmag[:-2, 1:-1]) & (c >= padmag[2:, 1:-1])
        survive_135 = (c >= padmag[:-2, :-2]) & (c >= padmag[2:, 2:])

        nms = torch.where(d0, survive_0.float() * mag, nms)
        nms = torch.where(d45, survive_45.float() * mag, nms)
        nms = torch.where(d90, survive_90.float() * mag, nms)
        nms = torch.where(d135, survive_135.float() * mag, nms)

        # 4. Double threshold + simple hysteresis
        strong = nms >= self.high_threshold
        weak = (nms >= self.low_threshold) & (~strong)

        # Dilate strong edges and include weak pixels adjacent to them
        strong_dilated = F.max_pool2d(
            strong.float().unsqueeze(0).unsqueeze(0),
            kernel_size=3, stride=1, padding=1,
        )[0, 0] > 0.5
        edges = strong | (weak & strong_dilated)

        return edges


# ---------------------------------------------------------------------------
# Pixel Assigner — Canny-guided colour matching
# ---------------------------------------------------------------------------

class PixelAssigner:
    """
    Assigns every source pixel to a unique target position.

    Edge target pixels are matched first (structure-critical), then the
    remaining pixels are matched by luminance-sorted pairing (fast OT
    approximation that preserves colour fidelity).

    Args:
        canny_sigma:          Gaussian sigma for Canny.
        canny_low:            Canny low threshold.
        canny_high:           Canny high threshold.
        edge_batch:           Batch size for edge matching to limit VRAM.
    """

    def __init__(
        self,
        canny_sigma: float = 1.4,
        canny_low: float = 0.05,
        canny_high: float = 0.15,
        edge_batch: int = 2048,
    ):
        self.canny = CannyEdgeDetector(canny_sigma, canny_low, canny_high)
        self.edge_batch = edge_batch

    @torch.no_grad()
    def assign(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute assignment: for every flat target index, which flat source
        index should provide its pixel?

        Args:
            source: (1, 3, H, W) source image tensor in [0, 1].
            target: (1, 3, H, W) target image tensor in [0, 1].

        Returns:
            (H*W,) long tensor of source indices (one per target pixel).
        """
        device = source.device
        _, _, h, w = source.shape
        n = h * w

        src_flat = source[0].permute(1, 2, 0).reshape(n, 3)   # (N, 3)
        tgt_flat = target[0].permute(1, 2, 0).reshape(n, 3)   # (N, 3)

        # Detect edges on target
        edge_mask = self.canny.detect(target)                  # (H, W) bool
        edge_flat = edge_mask.reshape(n)

        edge_indices = torch.where(edge_flat)[0]
        fill_indices = torch.where(~edge_flat)[0]

        assignment = torch.full((n,), -1, dtype=torch.long, device=device)
        used = torch.zeros(n, dtype=torch.bool, device=device)

        # --- Phase 1: match edge target pixels by colour distance ---
        if edge_indices.numel() > 0:
            tgt_edge = tgt_flat[edge_indices]                  # (E, 3)
            # Process in batches to control memory
            for start in range(0, edge_indices.numel(), self.edge_batch):
                end = min(start + self.edge_batch, edge_indices.numel())
                batch_tgt = tgt_edge[start:end]                # (B, 3)
                available = torch.where(~used)[0]
                src_avail = src_flat[available]                 # (A, 3)

                # Pairwise L2 distance  (B, A)
                dists = torch.cdist(batch_tgt, src_avail, p=2)
                # Greedy assignment: pick closest unused source for each target
                for i in range(batch_tgt.shape[0]):
                    best = dists[i].argmin()
                    src_idx = available[best]
                    tgt_idx = edge_indices[start + i]
                    assignment[tgt_idx] = src_idx
                    used[src_idx] = True
                    # Remove this source from future candidates in this batch
                    dists[:, best] = float("inf")

        # --- Phase 2: match remaining target pixels by sorted luminance ---
        remaining_tgt = fill_indices
        remaining_src = torch.where(~used)[0]

        if remaining_tgt.numel() > 0 and remaining_src.numel() > 0:
            tgt_lum = (
                0.2989 * tgt_flat[remaining_tgt, 0]
                + 0.5870 * tgt_flat[remaining_tgt, 1]
                + 0.1140 * tgt_flat[remaining_tgt, 2]
            )
            src_lum = (
                0.2989 * src_flat[remaining_src, 0]
                + 0.5870 * src_flat[remaining_src, 1]
                + 0.1140 * src_flat[remaining_src, 2]
            )
            tgt_order = tgt_lum.argsort()
            src_order = src_lum.argsort()

            # Pair the k-th darkest target pixel with the k-th darkest source
            count = min(tgt_order.numel(), src_order.numel())
            assignment[remaining_tgt[tgt_order[:count]]] = remaining_src[src_order[:count]]

        # Safety: any unassigned targets get a fallback
        unassigned = (assignment < 0)
        if unassigned.any():
            leftover_src = torch.where(~used)[0]
            unas_idx = torch.where(unassigned)[0]
            count = min(unas_idx.numel(), leftover_src.numel())
            assignment[unas_idx[:count]] = leftover_src[:count]
            if count < unas_idx.numel():
                assignment[unas_idx[count:]] = 0

        return assignment


# ---------------------------------------------------------------------------
# Physics Simulator — spring-damper particle system
# ---------------------------------------------------------------------------

class PhysicsSimulator:
    """
    GPU particle simulation: spring-damper dynamics driving source pixels
    toward their assigned target positions.

    Each of the H*W particles has:
        - position (x, y) in normalised pixel coords [0, W) × [0, H)
        - velocity (vx, vy)
        - colour   (r, g, b)
        - target   (tx, ty)

    The spring force is  F = -k * (pos - target) - damping * vel

    Args:
        source:      (1, 3, H, W) source image.
        assignment:  (H*W,) long tensor — target flat index per source pixel.
        device:      Torch device.
        spring_k:    Spring constant.
        damping:     Velocity damping coefficient.
        dt:          Timestep.
    """

    def __init__(
        self,
        source: torch.Tensor,
        assignment: torch.Tensor,
        device: torch.device,
        spring_k: float = 4.0,
        damping: float = 3.0,
        dt: float = 0.08,
    ):
        _, _, h, w = source.shape
        self.h = h
        self.w = w
        self.device = device
        self.spring_k = spring_k
        self.damping = damping
        self.dt = dt
        n = h * w

        # Pixel colours (fixed)
        self.colors = (
            source[0].detach().to(device).permute(1, 2, 0).reshape(n, 3)
        )

        # Source positions (starting positions) — normalised to pixel coords
        ys = torch.arange(h, dtype=torch.float32, device=device)
        xs = torch.arange(w, dtype=torch.float32, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        self.positions = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)  # (N, 2)

        # Target positions derived from assignment
        assignment = assignment.to(device)
        tgt_y = (assignment // w).float()
        tgt_x = (assignment % w).float()
        self.targets = torch.stack([tgt_x, tgt_y], dim=1)  # (N, 2)

        # Velocities start at zero
        self.velocities = torch.zeros_like(self.positions)

    def step(self):
        """Advance one physics timestep."""
        force = -self.spring_k * (self.positions - self.targets) - self.damping * self.velocities
        self.velocities = self.velocities + force * self.dt
        self.positions = self.positions + self.velocities * self.dt

    def render(self) -> torch.Tensor:
        """
        Render current particle positions to an image via bilinear splatting.

        Returns:
            (1, 3, H, W) image tensor.
        """
        h, w = self.h, self.w
        n, channels = self.colors.shape

        x = self.positions[:, 0]
        y = self.positions[:, 1]

        x0 = torch.floor(x).long()
        y0 = torch.floor(y).long()
        x1 = x0 + 1
        y1 = y0 + 1

        wx1 = (x - x0.float()).unsqueeze(1)
        wy1 = (y - y0.float()).unsqueeze(1)
        wx0 = 1.0 - wx1
        wy0 = 1.0 - wy1

        accum = torch.zeros(h * w, channels, device=self.device)
        weight = torch.zeros(h * w, 1, device=self.device)

        def _splat(xi, yi, wi):
            valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            idx = (yi * w + xi).clamp(0, h * w - 1)
            v = valid.unsqueeze(1).float()
            accum.scatter_add_(0, idx.unsqueeze(1).expand(-1, channels), self.colors * wi * v)
            weight.scatter_add_(0, idx.unsqueeze(1), wi * v)

        _splat(x0, y0, wx0 * wy0)
        _splat(x1, y0, wx1 * wy0)
        _splat(x0, y1, wx0 * wy1)
        _splat(x1, y1, wx1 * wy1)

        # Normalise and fill holes with nearest-available colour
        mask = weight > 1e-6
        result = torch.zeros_like(accum)
        result[mask.expand_as(result)] = (accum / weight.clamp_min(1e-6))[mask.expand_as(result)]

        return result.view(h, w, channels).permute(2, 0, 1).unsqueeze(0)

    def convergence(self) -> float:
        """Fraction of pixels within 0.5px of their target."""
        dist = (self.positions - self.targets).norm(dim=1)
        return (dist < 0.5).float().mean().item()
