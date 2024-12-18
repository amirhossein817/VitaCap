import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from model import ImageCaptioningModel
import pickle
import argparse
from PIL import Image
import os


# Custom Dataset Class
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


def train(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model = model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for images, captions in dataloader:
            images = images.to(device)
            captions = captions.to(device)

            # Forward pass
            outputs = model(images)
            outputs = outputs.view(-1, outputs.size(-1))  # Reshape for loss computation
            targets = captions.view(-1)  # Flatten captions

            # Compute loss
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Image Captioning Model")
    parser.add_argument(
        "--image_folder", type=str, required=True, help="Path to image folder"
    )
    parser.add_argument(
        "--caption_file", type=str, required=True, help="Path to caption text file"
    )
    parser.add_argument(
        "--vocab_path", type=str, required=True, help="Path to vocabulary .pkl file"
    )
    parser.add_argument(
        "--swin_ckpt",
        type=str,
        required=True,
        help="Path to Swin Transformer checkpoint",
    )
    parser.add_argument(
        "--yolo_ckpt", type=str, required=True, help="Path to YOLO checkpoint"
    )
    parser.add_argument(
        "--rcnn_ckpt", type=str, required=True, help="Path to Faster R-CNN checkpoint"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer"
    )
    args = parser.parse_args()

    # Load Vocabulary
    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)

    # Image Transformations
    transform = transforms.Compose(
        [
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Dataset and DataLoader
    dataset = CaptionDataset(args.image_folder, args.caption_file, vocab, transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model Initialization
    model = ImageCaptioningModel(
        swin_checkpoint=args.swin_ckpt,
        yolo_checkpoint=args.yolo_ckpt,
        rcnn_checkpoint=args.rcnn_ckpt,
        vocab_path=args.vocab_path,
    )

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab("<pad>"))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train the Model
    train(
        model,
        dataloader,
        criterion,
        optimizer,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_epochs=args.num_epochs,
    )
