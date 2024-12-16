import torch
import torch.nn as nn


class DecoderModule(nn.Module):
    def __init__(self, input_channels, target_channels, output_size):
        """
        A basic decoder module that upsamples and refines the input features.

        Args:
            input_channels (int): Number of input channels from the encoder output.
            target_channels (int): Desired number of output channels.
            output_size (tuple): Spatial dimensions of the final output (height, width).
        """
        super(DecoderModule, self).__init__()
        self.upsample = nn.Upsample(
            size=output_size, mode="bilinear", align_corners=True
        )
        self.conv1 = nn.Conv2d(
            input_channels, target_channels, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            target_channels, target_channels, kernel_size=3, padding=1
        )
        self.norm = nn.LayerNorm([target_channels, output_size[0], output_size[1]])
        self.activation = nn.ReLU()

    def forward(self, encoded_features):
        """
        Forward pass of the decoder.

        Args:
            encoded_features (torch.Tensor): Input feature map from the encoder.

        Returns:
            torch.Tensor: Refined and upsampled feature map.
        """
        x = self.upsample(encoded_features)  # Upsample to the target size
        x = self.conv1(x)  # First convolution
        x = self.activation(x)  # Activation
        x = self.conv2(x)  # Second convolution
        x = self.activation(x)  # Activation
        x = self.norm(x)  # Layer normalization
        return x


# Example usage
if __name__ == "__main__":
    decoder = DecoderModule(
        input_channels=256, target_channels=3, output_size=(224, 224)
    )

    # Dummy input feature map
    encoded_features = torch.randn(1, 256, 20, 20)

    # Decode the features
    decoded_output = decoder(encoded_features)
    print(f"Decoded Output Shape: {decoded_output.shape}")
