import torch
import os
import requests
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision import transforms
from PIL import Image
from src.models.feature_extractor import FeatureExtractor

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

        backbone = resnet_fpn_backbone('resnet101', pretrained=False)
        self.model = FasterRCNN(backbone, num_classes=91)  # 91 classes for COCO dataset

        # Load model weights from the checkpoint
        checkpoint = torch.load(self.checkpoint_path)
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint  # In case the file is only the state_dict

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def extract_features(self, image_path):
        """Extract features from the specified layer in the Faster R-CNN model."""
        if self.model is None:
            raise ValueError("Model is not loaded. Please call load_model() first.")

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        features = []

        def hook(module, input, output):
            features.append(output)

        # Register a hook on the layer of interest (adjust based on model structure)
        target_layer = self.model.backbone.body.layer3  # Example: the final ResNet layer
        hook_handle = target_layer.register_forward_hook(hook)

        # Preprocess the image
        input_image = Image.open(image_path).convert("RGB")
        input_tensor = transform(input_image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            _ = self.model(input_tensor)

        # Remove the hook after extracting features
        hook_handle.remove()

        if not features:
            raise RuntimeError("Failed to extract features. Check the target layer and image input.")

        return features[0]
