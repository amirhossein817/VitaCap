import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class CaptionDataset(Dataset):
    def __init__(self, image_folder, caption_file, vocab, transform=None):
        """
        Args:
            image_folder (str): Path to the folder containing images.
            caption_file (str): Path to the caption file (text file).
            vocab (Vocabulary): Vocabulary object.
            transform: Image preprocessing transformations.
        """
        self.image_folder = image_folder
        self.vocab = vocab
        self.transform = transform
        self.data = []

        # Load captions and corresponding image paths
        with open(caption_file, "r") as f:
            for line in f:
                image_file, caption = line.strip().split("\t")
                self.data.append((image_file, caption))

    def __getitem__(self, idx):
        image_path, caption = self.data[idx]
        image = Image.open(os.path.join(self.image_folder, image_path)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Convert caption to word indices
        tokens = caption.lower().split()
        caption_idx = (
            [self.vocab("<start>")]
            + [self.vocab(token) for token in tokens]
            + [self.vocab("<end>")]
        )

        return image, torch.tensor(caption_idx)

    def __len__(self):
        return len(self.data)
