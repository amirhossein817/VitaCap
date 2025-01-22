import torch
import os
from torchvision import transforms
from src.models.feature_extractor import FeatureExtractor
from ultralytics import YOLO


class YoloFeatureExtractor(FeatureExtractor):
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.model = None

    def load_model(self):
        model_name = os.path.splitext(os.path.basename(self.checkpoint_path))[0]
        # Check if the model file exists locally
        if os.path.exists(self.checkpoint_path):
            self.model = YOLO(self.checkpoint_path)
        else:
            self.model = YOLO(model_name)  # Download the model
            os.makedirs(
                os.path.dirname(self.checkpoint_path), exist_ok=True
            )  # Ensure the directory exists
            self.model.save(
                self.checkpoint_path
            )  # Save the model to the specified path
        self.model.eval()

    def extract_features(self, model, image_tensor):
        """
        Extract features from the input image tensor.

        Args:
            model (YOLO): The YOLO model.
            image_tensor (torch.Tensor): Preprocessed image tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Extracted features of shape [B, C, H', W'].
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Please call load_model() first.")

        # Ensure the input tensor is on the correct device
        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device)

        # Forward pass through the YOLO model's initial layers (stem)
        with torch.no_grad():
            # Access the stem (initial layers)
            stem = model.model.model[0]
            features = stem(image_tensor)  # Pass through the stem

            # Access the feature extraction layers (e.g., the first CSPLayer)
            feature_extractor = model.model.model[1]  # Adjust this index as needed
            features = feature_extractor(features)  # Extract features

        # Resize features to the target size (20x20)
        features = torch.nn.functional.interpolate(
            features, size=(20, 20), mode="bilinear"
        )

        return features
