import torch
import os
import yaml
import pickle
from src.models.model import ImageCaptioningModel
from src.utils import generate_caption_masks
from src.data.dataset import ImageCaptioningDataset
from PIL import Image
import torchvision.transforms as transforms
from src.data.build_vocab import Vocabulary
from src.train.train import train_model
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import yaml
import pickle
from src.data.dataset import ImageCaptioningDataset
from src.models.model import ImageCaptioningModel
from src.utils import generate_caption_masks
from src.train.eval import evaluate_model  # Import the evaluation function
from src.data.data_loader import get_loader


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


# Load configuration
config_path = "/content/VitaCap/src/configs/config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Hyperparameters and paths
batch_size = config["training"]["batch_size"]
num_epochs = config["training"]["num_epochs"]
learning_rate = config["training"]["learning_rate"]
embed_size = config["model"]["embed_size"]
num_heads = config["model"]["num_heads"]
num_workers = config["model"]["worker"]
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
# train_data_path = config["dataset"]["train_data_path"]
val_data_path = config["dataset"]["val_data_path"]
save_path = config["paths"]["save_path"]

# Dataset
train_data_path = config["dataset"]["train_data_path"]
val_data_path = config["dataset"]["val_data_path"]
train_data_annotaions = config["dataset"]["train_data_annotaions"]
val_data_annotaions = config["dataset"]["train_data_annotaions"]

os.makedirs(save_path, exist_ok=True)

# Load vocabulary
with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)

image_size = config["transform"]["size"]
transform = transforms.Compose(
    [
        transforms.Pad((0, 0, image_size, image_size), fill=0, padding_mode="constant"),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

# # Load dataset
# train_dataset = ImageCaptioningDataset(
#     train_data_path, vocab_path, max_seq_length=max_seq_length
# )
# val_dataset = ImageCaptioningDataset(
#     val_data_path, vocab_path, max_seq_length=max_seq_length
# )

# train_loader = DataLoader(
#     train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
# )
# val_loader = DataLoader(
#     val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
# )

# Build data loader
data_loader_train = get_loader(
    train_data_path,
    train_data_annotaions,
    vocab,
    transform,
    batch_size,
    shuffle=True,
    num_workers=num_workers,
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
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Train the model
train_model(
    model=model,
    train_loader=data_loader_train,
    val_loader=None,
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
