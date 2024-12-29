import torch
import os
import yaml
from src.models.model import ImageCaptioningModel
from src.utils import generate_caption_masks
from src.data.dataset import ImageCaptioningDataset
from PIL import Image
import torchvision.transforms as transforms


def load_config(config_path):
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def preprocess_image(image_path):
    """
    Preprocess the input image for inference.

    Args:
        image_path (str): Path to the input image.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension


def generate_caption(model, image, vocab, max_seq_length, device):
    """
    Generate a caption for the input image.

    Args:
        model (ImageCaptioningModel): Trained image captioning model.
        image (torch.Tensor): Preprocessed image tensor.
        vocab (Vocabulary): Vocabulary object for decoding indices.
        max_seq_length (int): Maximum sequence length for captions.
        device (torch.device): Device to perform inference on.

    Returns:
        str: Generated caption.
    """
    model.eval()
    image = image.to(device)

    # Start token
    caption = [vocab.word2idx["<start>"]]

    for _ in range(max_seq_length):
        caption_tensor = torch.tensor(caption, dtype=torch.long).unsqueeze(0).to(device)
        tgt_mask = generate_caption_masks(len(caption)).to(device)

        with torch.no_grad():
            output = model(image, caption_tensor, tgt_mask)

        # Get the next word
        next_word_idx = torch.argmax(output[0, -1, :]).item()
        caption.append(next_word_idx)

        # Stop if end token is generated
        if next_word_idx == vocab.word2idx["<end>"]:
            break

    # Convert indices to words
    caption_words = [
        vocab.idx2word[idx]
        for idx in caption
        if idx not in [vocab.word2idx["<start>"], vocab.word2idx["<end>"]]
    ]
    return " ".join(caption_words)


if __name__ == "__main__":
    # Load configuration
    config_path = "config.yaml"
    config = load_config(config_path)

    # Hyperparameters and paths
    swin_ckpt = config["paths"]["swin_checkpoint"]
    yolo_ckpt = config["paths"]["yolo_checkpoint"]
    rcnn_ckpt = config["paths"]["rcnn_checkpoint"]
    vocab_path = config["paths"]["vocab_path"]
    model_checkpoint = config["paths"]["model_checkpoint"]
    max_seq_length = config["model"]["max_seq_length"]
    image_path = "./image.jpg"  # Replace with your test image path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocabulary
    with open(vocab_path, "rb") as f:
        vocab = torch.load(f)

    # Instantiate model
    model = ImageCaptioningModel(
        swin_checkpoint=swin_ckpt,
        yolo_checkpoint=yolo_ckpt,
        rcnn_checkpoint=rcnn_ckpt,
        vocab_path=vocab_path,
        embed_size=config["model"]["embed_size"],
        num_heads=config["model"]["num_heads"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        target_channels=config["model"]["target_channels"],
        target_size=tuple(config["model"]["target_size"]),
        max_seq_length=max_seq_length,
    )

    # Load model weights
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model = model.to(device)

    # Preprocess input image
    image = preprocess_image(image_path)

    # Generate caption
    caption = generate_caption(model, image, vocab, max_seq_length, device)
    print("Generated Caption:", caption)
