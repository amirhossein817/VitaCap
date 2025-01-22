import torch
import os
from torchvision import transforms
from PIL import Image
from src.models.feature_extractor import FeatureExtractor
from src.models.swin_transformer import SwinTransformerV2


class SwinFeatureExtractor(FeatureExtractor):
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.model = None

    def load_model(self):
        if not os.path.exists(self.checkpoint_path):
            model_name = os.path.splitext(os.path.basename(self.checkpoint_path))[0]
            self.model = SwinTransformerV2(
                img_size=384,
                embed_dim=192,
                depths=[2, 2, 18, 2],
                num_heads=[6, 12, 24, 48],
                window_size=12,
                num_classes=1000,
            )
            torch.save(self.model.state_dict(), self.checkpoint_path)
        else:
            self.model = SwinTransformerV2(
                img_size=384,
                embed_dim=192,
                depths=[2, 2, 18, 2],
                num_heads=[6, 12, 24, 48],
                window_size=12,
                num_classes=1000,
            )
            self.model.load_weights(self.checkpoint_path)
        self.model.eval()

    def extract_features(self, image_tensor):
        """
        Extract features from the input image tensor.

        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Extracted features of shape [B, C, H', W'].
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Please call load_model() first.")

        # Ensure the input tensor is on the correct device
        device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(device)

        # Forward pass through the Swin Transformer
        with torch.no_grad():
            features = self.model(image_tensor)  # Shape: [B, L, C]

        # Reshape features to [B, C, H, W]
        batch_size, seq_len, channels = features.shape
        height = width = int(seq_len**0.5)  # Assuming L = H * W
        features = features.permute(0, 2, 1).reshape(
            batch_size, channels, height, width
        )

        # Interpolate features to the target size (20x20)
        features = torch.nn.functional.interpolate(
            features, size=(20, 20), mode="bilinear"
        )

        return features
