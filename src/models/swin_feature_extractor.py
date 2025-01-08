import torch
import os
from torchvision import transforms
from PIL import Image
from transformers import SwinModel, SwinConfig
from src.models.swin_transformer import SwinTransformerV2
from src.models.feature_extractor import FeatureExtractor


class SwinFeatureExtractor(FeatureExtractor):
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.model = None

    def load_model(self):

        if not os.path.exists(self.checkpoint_path):
            model_name = os.path.splitext(os.path.basename(self.checkpoint_path))[0]
            config = SwinConfig.from_pretrained(f"microsoft/{model_name}")
            swin = SwinModel.from_pretrained(f"microsoft/{model_name}", config=config)
            torch.save(swin.state_dict(), self.checkpoint_path)

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

    def extract_features(self, image_path):
        """Extract features from the input image."""
        preprocess = transforms.Compose(
            [
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        if self.model is None:
            raise ValueError("Model is not loaded. Please call load_model() first.")

        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")

        # print(type(image_path))
        # if isinstance(image_path, str):  # If input is a file path
        #   image = Image.open(image_path).convert("RGB")
        #   image = preprocess(image).unsqueeze(0)  # Add batch dimension
        # elif isinstance(image, torch.Tensor):
        #   if len(image.shape) == 3:  # Add batch dimension if missing
        #     image = image.unsqueeze(0)
        # else:
        #   raise ValueError("Input must be a file path or a preprocessed tensor.")

        input = preprocess(image).unsqueeze(0)  # Add batch dimension
        output = self.model(input)
        output = output.view(1, 144, 32, 48)  # reshape to pseudo 2D
        output = torch.nn.functional.interpolate(output, size=(20, 20), mode="bilinear")
        output = output.permute(0, 2, 3, 1).reshape(1, -1, 20, 20)
        return output
