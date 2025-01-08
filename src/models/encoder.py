import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.fusion_module import FeatureFusion


class EncoderModule(nn.Module):
    def __init__(self, target_channels=512, target_size=(20, 20), device=None):
        super(EncoderModule, self).__init__()
        self.target_channels = target_channels
        self.target_size = target_size
        if device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        # Feature fusion module
        self.feature_fusion_left = FeatureFusion(target_channels, target_size).to(
            self.device
        )
        self.feature_fusion_right = FeatureFusion(target_channels, target_size).to(
            self.device
        )
        self.feature_fusion_final = FeatureFusion(target_channels, target_size).to(
            self.device
        )

        # Additional layers for encoding
        self.conv_final = nn.Conv2d(
            target_channels, target_channels, kernel_size=3, padding=1
        )
        self.norm_final = nn.LayerNorm(
            [target_channels, target_size[0], target_size[1]]
        )

    def forward(self, feature_swin, feature_yolo, feature_rcnn):
        """
        Encoder forward pass that fuses three input feature maps.

        Args:
            feature_swin (torch.Tensor): Swin Transformer feature map.
            feature_yolo (torch.Tensor): YOLO feature map.
            feature_rcnn (torch.Tensor): Faster R-CNN feature map.

        Returns:
            torch.Tensor: Encoded fused feature map.
        """
        # Fuse Swin and YOLO features
        fused1 = self.feature_fusion_left(feature_swin, feature_yolo).to(self.device)

        # Fuse Swin and Faster R-CNN features
        fused2 = self.feature_fusion_right(feature_swin, feature_rcnn).to(self.device)

        # Fuse the results of the previous fusions
        fused_output = self.feature_fusion_final(fused1, fused2)

        # Apply final convolution and normalization
        fused_output = self.conv_final(fused_output)
        fused_output = self.norm_final(fused_output)

        return fused_output
