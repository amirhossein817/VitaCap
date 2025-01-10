import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import yaml
import pickle
from src.data.dataset import ImageCaptioningDataset
from src.models.model import ImageCaptioningModel
from src.utils import generate_caption_masks
from eval import evaluate_model  # Import the evaluation function


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


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    device,
    save_path,
    vocab,
    max_seq_length,
    dataset_type,
):
    """
    Train the image captioning model and evaluate after each epoch.

    Args:
        model (nn.Module): The image captioning model.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        num_epochs (int): Number of epochs to train for.
        device (torch.device): Device to train on (e.g., 'cuda' or 'cpu').
        save_path (str): Path to save the trained model checkpoint.
        vocab (Vocabulary): Vocabulary object.
        max_seq_length (int): Maximum sequence length for captions.
        dataset_type (str): Dataset type, either 'coco' or 'flickr'.

    Returns:
        None
    """
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, captions in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):
            images = images.to(device)
            captions = captions.to(device)

            # Generate masks for the decoder
            tgt_mask = generate_caption_masks(captions.size(1)).to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images, captions[:, :-1], tgt_mask)

            # Calculate loss
            loss = criterion(
                outputs.view(-1, outputs.size(-1)), captions[:, 1:].reshape(-1)
            )

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Adjust learning rate
        scheduler.step()

        # Validation and evaluation
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, captions in val_loader:
                images = images.to(device)
                captions = captions.to(device)

                tgt_mask = generate_caption_masks(captions.size(1)).to(device)

                outputs = model(images, captions[:, :-1], tgt_mask)
                loss = criterion(
                    outputs.view(-1, outputs.size(-1)), captions[:, 1:].reshape(-1)
                )

                val_loss += loss.item()

        # Evaluate metrics
        metrics = evaluate_model(
            model, val_loader, vocab, device, max_seq_length, dataset_type
        )

        # Logging
        print(
            f"Epoch [{epoch + 1}/{num_epochs}]: "
            f"Train Loss: {train_loss / len(train_loader):.4f}, "
            f"Val Loss: {val_loss / len(val_loader):.4f}"
        )
        print("Evaluation Metrics:")
        for metric, score in metrics.items():
            print(f"{metric}: {score:.4f}")

        # Save the model
        torch.save(
            model.state_dict(), os.path.join(save_path, f"model_epoch_{epoch + 1}.pth")
        )


if __name__ == "__main__":
    # Load configuration
    config_path = "config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Hyperparameters and paths
    batch_size = config["training"]["batch_size"]
    num_epochs = config["training"]["num_epochs"]
    learning_rate = config["training"]["learning_rate"]
    embed_size = config["model"]["embed_size"]
    num_heads = config["model"]["num_heads"]
    hidden_dim = config["model"]["hidden_dim"]
    num_layers = config["model"]["num_layers"]
    target_channels = config["model"]["target_channels"]
    target_size = tuple(config["model"]["target_size"])
    max_seq_length = config["model"]["max_seq_length"]
    dataset_type = config["dataset"]["type"]  # Read dataset type from config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    swin_ckpt = config["paths"]["swin_checkpoint"]
    yolo_ckpt = config["paths"]["yolo_checkpoint"]
    rcnn_ckpt = config["paths"]["rcnn_checkpoint"]
    vocab_path = config["paths"]["vocab_path"]
    train_data_path = config["dataset"]["train_data_path"]
    val_data_path = config["dataset"]["val_data_path"]
    save_path = config["paths"]["save_path"]

    os.makedirs(save_path, exist_ok=True)

    # Load vocabulary
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    # Load dataset
    train_dataset = ImageCaptioningDataset(
        train_data_path, vocab_path, max_seq_length=max_seq_length
    )
    val_dataset = ImageCaptioningDataset(
        val_data_path, vocab_path, max_seq_length=max_seq_length
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Instantiate model
    model = ImageCaptioningModel(
        swin_checkpoint=swin_ckpt,
        yolo_checkpoint=yolo_ckpt,
        rcnn_checkpoint=rcnn_ckpt,
        vocab_path=vocab_path,
        embed_size=embed_size,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        target_channels=target_channels,
        target_size=target_size,
        max_seq_length=max_seq_length,
    )

    # Loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=device,
        save_path=save_path,
        vocab=vocab,
        max_seq_length=max_seq_length,
        dataset_type=dataset_type,  # Pass dataset type
    )