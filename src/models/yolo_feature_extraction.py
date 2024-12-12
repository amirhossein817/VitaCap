import torch
import os
from torchvision import transforms
from PIL import Image
from src.models.feature_extractor import FeatureExtractor
from torchvision import transforms
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

    # Define a feature extraction function
    def extract_features(self, model, image_path):
        if self.model is None:
            raise ValueError("Model is not loaded. Please call load_model() first.")
        transform = transforms.Compose(
            [
                transforms.Resize((384, 384)),  # Resize to model's input size
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # ImageNet normalization
            ]
        )
        features = []

        def hook(module, input, output):
            features.append(output)

        target_layer = model.model.model[10]  # Adjust the index to the desired layer
        hook_handle = target_layer.register_forward_hook(hook)
        input_image = Image.open(image_path).convert("RGB")
        input_tensor = transform(input_image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            _ = model(input_tensor)
        extracted_features = features[0]
        hook_handle.remove()
        return extracted_features
