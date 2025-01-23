import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusion(nn.Module):
    def __init__(self, target_channels, target_size):
        super(FeatureFusion, self).__init__()
        self.target_channels = target_channels
        self.target_size = target_size

        # Use ModuleDict to store 1x1 convolution layers for dynamic alignment
        self.conv_layers = nn.ModuleDict()

    def align_features(self, features, input_name):
        """
        Align the input feature dimensions to the target dimensions.

        Args:
            features (torch.Tensor): Input feature map [batch_size, channels, height, width]
            input_name (str): A unique identifier for the input feature map (e.g., "swin", "yolo").

        Returns:
            torch.Tensor: Aligned feature map with target channels and spatial dimensions.
        """
        batch_size, channels, height, width = features.size()

        # Dynamically create a 1x1 convolution layer if needed
        if input_name not in self.conv_layers:
            self.conv_layers[input_name] = nn.Conv2d(
                channels, self.target_channels, kernel_size=1
            )

        self.conv_layers[input_name] = self.conv_layers[input_name].to(features.device)
        # Apply 1x1 convolution for channel alignment
        aligned_features = self.conv_layers[input_name](features)

        # Apply bilinear interpolation for spatial alignment
        if (height, width) != self.target_size:
            aligned_features = F.interpolate(
                aligned_features, size=self.target_size, mode="bilinear"
            )

        return aligned_features

    def forward(self, feature1, feature2):
        """
        Combine two input feature maps with Multi-Head Attention.

        Args:
            feature1 (torch.Tensor): First input feature map.
            feature2 (torch.Tensor): Second input feature map.

        Returns:
            torch.Tensor: Fused feature map.
        """
        # print(feature1.shape)
        # print(feature2.shape)
        # Align both input features
        feature1 = self.align_features(feature1, "feature1")
        feature2 = self.align_features(feature2, "feature2")
        device = feature1.device
        # Multi-Head Attention setup (example, replace with your implementation)
        mha1 = nn.MultiheadAttention(embed_dim=self.target_channels, num_heads=8).to(
            device
        )
        mha2 = nn.MultiheadAttention(embed_dim=self.target_channels, num_heads=8).to(
            device
        )

        # Reshape for attention input (batch_first=True)
        batch_size, channels, height, width = feature1.size()
        feature1_flatten = feature1.view(batch_size, channels, -1).permute(
            0, 2, 1
        )  # [batch, seq_len, channels]
        feature2_flatten = feature2.view(batch_size, channels, -1).permute(0, 2, 1)
        # Move features to the correct device
        feature1_flatten = feature1_flatten.to(device)
        feature2_flatten = feature2_flatten.to(device)
        # Attention
        attn1, _ = mha1(feature1_flatten, feature2_flatten, feature2_flatten)
        attn2, _ = mha2(feature2_flatten, feature1_flatten, feature1_flatten)

        # Combine the outputs
        fused_features = attn1 + attn2
        fused_features = fused_features.permute(0, 2, 1).view(
            batch_size, self.target_channels, height, width
        )

        # Final Add & Norm
        fused_features = F.layer_norm(fused_features, fused_features.size()[1:])

        return fused_features


# Example usage
if __name__ == "__main__":
    feature_fusion = FeatureFusion(target_channels=256, target_size=(20, 20))

    # Dummy input features
    feature1 = torch.randn(1, 144, 20, 20)
    feature2 = torch.randn(1, 768, 15, 15)

    fused_output = feature_fusion(feature1, feature2)
    print(f"Fused Output Shape: {fused_output.shape}")
