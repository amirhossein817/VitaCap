import torch
import os
import requests
from torchvision import transforms
from src.models.feature_extractor import FeatureExtractor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class FasterRCNNFeatureExtractor(FeatureExtractor):
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.model = None

    def load_model(self):
        """Load the Faster R-CNN model with a ResNet-101 backbone."""
        model_url = "https://ababino-models.s3.amazonaws.com/resnet101_7a82fa4a.pth"

        # Download model if not already available
        if not os.path.exists(self.checkpoint_path):
            print("Downloading pretrained model...")
            response = requests.get(model_url)
            with open(self.checkpoint_path, "wb") as f:
                f.write(response.content)
            print("Model downloaded successfully.")

        backbone = resnet_fpn_backbone("resnet101", pretrained=False)
        self.model = FasterRCNN(backbone, num_classes=91)  # 91 classes for COCO dataset

        # Load model weights from the checkpoint
        checkpoint = torch.load(self.checkpoint_path)
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint  # In case the file is only the state_dict

        self.model.load_state_dict(state_dict)
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

        # Forward pass through the Faster R-CNN model
        with torch.no_grad():
            features = self.model.backbone(
                image_tensor
            )  # Extract features from the backbone

        # Resize features to the target size (20x20)
        features = torch.nn.functional.interpolate(
            features["0"], size=(20, 20), mode="bilinear"
        )

        return features
