import torch
import torch.nn as nn
import pickle
from src.models.swin_feature_extractor import SwinFeatureExtractor
from src.models.yolo_feature_extraction import YoloFeatureExtractor
from src.models.faster_rcnn_featuure_extractor import FasterRCNNFeatureExtractor
from src.models.encoder import EncoderModule
from src.models.decoder import DecoderModule


class ImageCaptioningModel(nn.Module):
    def __init__(
        self,
        swin_checkpoint,
        yolo_checkpoint,
        rcnn_checkpoint,
        vocab_path,
        embed_size=256,
        num_heads=8,
        hidden_dim=512,
        num_layers=6,
        target_channels=512,
        target_size=(20, 20),
        max_seq_length=50,
    ):
        """
        Unified Image Captioning Model.

        Args:
            swin_checkpoint (str): Path to Swin Transformer checkpoint.
            yolo_checkpoint (str): Path to YOLO checkpoint.
            rcnn_checkpoint (str): Path to Faster R-CNN checkpoint.
            vocab_path (str): Path to the vocabulary .pkl file.
            embed_size (int): Size of embeddings for the decoder.
            num_heads (int): Number of attention heads for the decoder.
            hidden_dim (int): Hidden dimension size for the decoder.
            num_layers (int): Number of layers in the decoder.
            target_channels (int): Number of target channels for fusion and encoder.
            target_size (tuple): Spatial dimensions for feature alignment.
            max_seq_length (int): Maximum sequence length for captions.
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
        vocab = self.load_vocab(vocab_path)
        vocab_size = len(vocab)
        self.decoder = DecoderModule(
            vocab_size=vocab_size,
            embed_size=embed_size,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_seq_length=max_seq_length,
        )

        # Load Vocabulary
        self.vocab = vocab

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

    def forward(self, image, captions, tgt_mask=None):
        """
        Forward pass through the model.

        Args:
            image (torch.Tensor): Input image tensor of shape [B, 3, H, W].
            captions (torch.Tensor): Input captions tensor of shape [B, seq_len].
            tgt_mask (torch.Tensor, optional): Target mask for decoder [seq_len, seq_len].

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

        # Decoder: Generate captions
        caption_logits = self.decoder(captions, encoded_features, tgt_mask=tgt_mask)

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
            " ".join([self.vocab.idx2word[idx.item()] for idx in seq])
            for seq in predicted_indices
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
    dummy_captions = torch.randint(0, 5000, (1, 50))  # Dummy captions

    # Forward pass
    logits = model(dummy_image, dummy_captions)
    print("Caption Logits Shape:", logits.shape)

    # Generate predicted caption
    caption = model.predict_caption(logits)
    print("Predicted Caption:", caption)
