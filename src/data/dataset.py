import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


class ImageCaptioningDataset(Dataset):
    def __init__(self, data_path, vocab_path, max_seq_length=50):
        """
        Initialize the dataset for image captioning.

        Args:
            data_path (str): Path to the dataset JSON file.
            vocab_path (str): Path to the vocabulary .pkl file.
            max_seq_length (int): Maximum sequence length for captions.
        """
        self.data_path = data_path
        self.max_seq_length = max_seq_length

        # Load data
        with open(data_path, "r") as file:
            self.data = json.load(file)

        # Load vocabulary
        with open(vocab_path, "rb") as f:
            self.vocab = torch.load(f)

        self.pad_idx = self.vocab.word2idx["<pad>"]
        self.start_idx = self.vocab.word2idx["<start>"]
        self.end_idx = self.vocab.word2idx["<end>"]

        # Define image transformations
        self.transforms = transforms.Compose(
            [
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get an image-caption pair.

        Args:
            idx (int): Index of the data point.

        Returns:
            tuple: Transformed image and tokenized caption.
        """
        item = self.data[idx]

        # Load and transform the image
        image_path = item["image"]
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)

        # Tokenize caption
        caption = item["caption"]
        tokens = (
            [self.start_idx]
            + [self.vocab(word) for word in caption.split()]
            + [self.end_idx]
        )
        if len(tokens) > self.max_seq_length:
            tokens = tokens[: self.max_seq_length - 1] + [self.end_idx]

        # Pad the sequence
        tokens += [self.pad_idx] * (self.max_seq_length - len(tokens))

        return image, torch.tensor(tokens, dtype=torch.long)


def generate_caption_masks(seq_len):
    """
    Generate target masks for the captions.

    Args:
        seq_len (int): Length of the target sequence.

    Returns:
        torch.Tensor: Target mask.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask


if __name__ == "__main__":
    # Example usage
    data_path = "path/to/dataset.json"
    vocab_path = "path/to/vocab.pkl"
    dataset = ImageCaptioningDataset(data_path, vocab_path)

    print("Dataset size:", len(dataset))

    # Example: Get one data point
    image, caption = dataset[0]
    print("Image shape:", image.shape)
    print("Caption tokens:", caption)
