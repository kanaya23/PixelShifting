"""
VGG19 Feature Extractor — Frozen perceptual feature extraction.

Extracts multi-layer features from VGG19 for computing perceptual loss.
All weights are frozen; ImageNet normalization is applied automatically.
"""

import torch
import torch.nn as nn
from torchvision import models


# ImageNet normalization constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# VGG19 layer name → index in vgg19.features sequential
VGG19_LAYER_MAP = {
    "relu1_2": 3,
    "relu2_2": 8,
    "relu3_4": 17,
    "relu4_4": 26,
    "relu5_4": 35,
}


class VGGFeatureExtractor(nn.Module):
    """
    Extracts features from selected VGG19 layers.

    Args:
        layers:  List of layer names to extract from.
                 Default: ['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']
        device:  Torch device to run the model on.

    Usage:
        extractor = VGGFeatureExtractor(device=torch.device('cuda:0'))
        features = extractor(image_tensor)  # dict of {layer_name: feature_tensor}
    """

    def __init__(
        self,
        layers: list = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device

        if layers is None:
            layers = ["relu1_2", "relu2_2", "relu3_4", "relu4_4", "relu5_4"]

        self.layer_names = layers
        self.layer_indices = {name: VGG19_LAYER_MAP[name] for name in layers}

        # Load pretrained VGG19 and extract only the feature layers we need
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        max_idx = max(self.layer_indices.values()) + 1
        self.features = nn.Sequential(*list(vgg.features.children())[:max_idx])

        # Freeze all parameters
        for param in self.features.parameters():
            param.requires_grad = False

        self.features.eval()
        self.features.to(device)

        # Pre-compute normalization tensors on the target device
        self.register_buffer(
            "mean", IMAGENET_MEAN.to(device)
        )
        self.register_buffer(
            "std", IMAGENET_STD.to(device)
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Extract features from the input image.

        Args:
            x: (N, 3, H, W) image tensor in [0, 1] range.

        Returns:
            Dictionary mapping layer names to feature tensors.
        """
        # Move to device if needed
        x = x.to(self.device)

        # Apply ImageNet normalization
        x = (x - self.mean) / self.std

        features = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            # Check if this index corresponds to one of our target layers
            for name, idx in self.layer_indices.items():
                if i == idx:
                    features[name] = x

        return features
