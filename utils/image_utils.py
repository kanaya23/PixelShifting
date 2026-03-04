"""
Image Utilities — Loading, saving, and tensor/QImage conversions.
"""

import numpy as np
import torch
from PIL import Image


def load_image(
    path: str,
    size: int = 256,
) -> torch.Tensor:
    """
    Load an image and resize to (size x size).

    Args:
        path: Path to the image file.
        size: Target resolution (images are resized to size x size).

    Returns:
        (1, 3, H, W) float32 tensor in [0, 1] range.
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3) in [0, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return tensor


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert (1, 3, H, W) tensor to (H, W, 3) uint8 numpy array.
    """
    img = tensor[0].detach().cpu().clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()  # (H, W, 3)
    return (img * 255).astype(np.uint8)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert (1, 3, H, W) tensor to PIL Image."""
    arr = tensor_to_numpy(tensor)
    return Image.fromarray(arr)


def tensor_to_qimage(tensor: torch.Tensor):
    """
    Convert (1, 3, H, W) tensor to QImage.

    Returns a QImage in RGB888 format.
    """
    from PyQt5.QtGui import QImage

    arr = tensor_to_numpy(tensor)
    h, w, c = arr.shape
    bytes_per_line = c * w
    # Ensure contiguous memory
    arr = np.ascontiguousarray(arr)
    return QImage(arr.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()


def save_image(tensor: torch.Tensor, path: str):
    """Save (1, 3, H, W) tensor as an image file."""
    img = tensor_to_pil(tensor)
    img.save(path)
