import torch
import torch.nn as nn
import pickle
from swin_feature_extractor import SwinFeatureExtractor
from yolo_feature_extraction import YoloFeatureExtractor
from faaster_rcnn_featuure_extractor import FasterRCNNFeatureExtractor
from encoder import EncoderModule
from decoder import DecoderModule


class ImageCaptioningModel(nn.Module):
    def __init__(
        self,
        swin_checkpoint,
        yolo_checkpoint,
        rcnn_checkpoint,
        vocab_path,
        target_channels=512,
        target_size=(20, 20),
        output_size=(224, 224),
    ):
        """
        Unified Image Captioning Model with Vocabulary Integration.

        Args:
            swin_checkpoint (str): Path to Swin Transformer checkpoint.
            yolo_checkpoint (str): Path to YOLO checkpoint.
            rcnn_checkpoint (str): Path to Faster R-CNN checkpoint.
            vocab_path (str): Path to the vocabulary .pkl file.
            target_channels (int): Number of target channels for fusion and encoder.
            target_size (tuple): Spatial dimensions for feature alignment.
            output_size (tuple): Spatial dimensions for the decoder output.
        """
        super(ImageCaptioningModel, self).__init__()

        # Feature Extractors
        self.swin_extractor = SwinFeatureExtractor(swin_checkpoint)
        self.yolo_extractor = YoloFeatureExtractor(yolo_checkpoint)
        self.rcnn_extractor = FasterRCNNFeatureExtractor(rcnn_checkpoint)

        # Encoder
        self.encoder = EncoderModule(
            target_channels=target_channels, target_size=target_size
        )

        # Decoder
        self.decoder = DecoderModule(
            input_channels=target_channels,
            target_channels=target_channels,
            output_size=output_size,
        )

        # Caption Generation Head
        self.caption_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(target_channels * output_size[0] * output_size[1], 1024),
            nn.ReLU(),
            nn.Linear(
                1024, len(self.load_vocab(vocab_path))
            ),  # Output size matches vocab size
        )

        # Load Vocabulary
        self.vocab = self.load_vocab(vocab_path)

        # Initialize feature extractors
        self.swin_extractor.load_model()
        self.yolo_extractor.load_model()
        self.rcnn_extractor.load_model()

    def load_vocab(self, vocab_path):
        """
        Load the vocabulary from a pickle file.
        Args:
            vocab_path (str): Path to the vocabulary .pkl file.
        Returns:
            vocab (Vocabulary): Loaded vocabulary object.
        """
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        return vocab

    def forward(self, image):
        """
        Forward pass through the model.

        Args:
            image (torch.Tensor): Input image tensor of shape [B, 3, H, W].

        Returns:
            torch.Tensor: Predicted caption logits.
        """
        # Feature Extraction
        swin_features = self.swin_extractor.extract_features(image)
        yolo_features = self.yolo_extractor.extract_features(
            self.yolo_extractor.model, image
        )
        rcnn_features = self.rcnn_extractor.extract_features(image)

        # Encoder: Fuse and encode features
        encoded_features = self.encoder(swin_features, yolo_features, rcnn_features)

        # Decoder: Refine features
        decoded_features = self.decoder(encoded_features)

        # Caption Generation
        caption_logits = self.caption_head(decoded_features)

        return caption_logits

    def predict_caption(self, logits):
        """
        Convert logits to words using the vocabulary.

        Args:
            logits (torch.Tensor): Predicted logits from the model.

        Returns:
            list[str]: List of predicted words.
        """
        predicted_indices = torch.argmax(logits, dim=-1)
        predicted_caption = [
            self.vocab.idx2word[idx.item()] for idx in predicted_indices[0]
        ]
        return predicted_caption


# Example Usage
if __name__ == "__main__":
    # Dummy paths for checkpoints and vocabulary
    swin_ckpt = "path/to/swin_checkpoint.pth"
    yolo_ckpt = "path/to/yolo_checkpoint.pt"
    rcnn_ckpt = "path/to/rcnn_checkpoint.pth"
    vocab_path = "path/to/vocab.pkl"

    # Instantiate model
    model = ImageCaptioningModel(swin_ckpt, yolo_ckpt, rcnn_ckpt, vocab_path)

    # Dummy input image (batch_size=1, 3 channels, 384x384 resolution)
    dummy_image = torch.randn(1, 3, 384, 384)

    # Forward pass
    logits = model(dummy_image)
    print("Caption Logits Shape:", logits.shape)

    # Generate predicted caption
    caption = model.predict_caption(logits)
    print("Predicted Caption:", " ".join(caption))
