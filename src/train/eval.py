import torch
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
import os
from tqdm import tqdm
from src.data.dataset import ImageCaptioningDataset
from src.utils import generate_caption_masks


def evaluate_model(model, dataloader, vocab, device, max_seq_length=50, dataset_type="coco"):
    """
    Evaluate the model on the validation dataset and compute metrics.

    Args:
        model (nn.Module): Trained image captioning model.
        dataloader (DataLoader): DataLoader for the validation dataset.
        vocab (Vocabulary): Vocabulary object.
        device (torch.device): Device to run evaluation on.
        max_seq_length (int): Maximum sequence length for captions.
        dataset_type (str): Dataset type, either 'coco' or 'flickr'.

    Returns:
        dict: Dictionary containing BLEU (1-4), CIDEr, ROUGE-L, and SPICE scores.
    """
    model.eval()
    results = []

    # Load ground truth annotations
    if dataset_type.lower() == "coco":
        coco = COCO(dataloader.dataset.data_path)  # Load COCO annotations
    elif dataset_type.lower() == "flickr":
        # For Flickr, create a dummy COCO object
        coco = create_flickr_coco(dataloader.dataset.data_path)
    else:
        raise ValueError("Unsupported dataset type. Use 'coco' or 'flickr'.")

    for images, image_ids in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)

        # Generate captions
        captions = generate_captions(model, images, vocab, max_seq_length, device)

        # Save results in COCO evaluation format
        for image_id, caption in zip(image_ids, captions):
            results.append(
                {
                    "image_id": image_id.item(),
                    "caption": caption,
                }
            )

    # Save results to a temporary JSON file
    results_file = "temp_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f)

    # Load results and ground truth for evaluation
    coco_res = coco.loadRes(results_file)
    coco_eval = COCOEvalCap(coco, coco_res)

    # Evaluate and get scores
    coco_eval.evaluate()
    metrics = coco_eval.eval

    # Clean up temporary file
    os.remove(results_file)

    return metrics


def create_flickr_coco(annotation_path):
    """
    Create a dummy COCO object for Flickr dataset.

    Args:
        annotation_path (str): Path to the Flickr annotation file.

    Returns:
        COCO: A COCO object with Flickr annotations.
    """
    with open(annotation_path, "r") as f:
        lines = f.readlines()

    # Create a dummy COCO annotation structure
    annotations = []
    images = []
    for idx, line in enumerate(lines):
        if len(line.strip()) == 0:
            continue
        # Split image ID and caption
        image_id, caption = line.strip().split("\t")
        annotations.append(
            {
                "image_id": int(image_id),
                "id": idx,
                "caption": caption,
            }
        )
        images.append(
            {
                "id": int(image_id),
                "file_name": f"{image_id}.jpg",  # Assuming image filenames match IDs
            }
        )

    # Create COCO object
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "object"}],  # Dummy category
    }
    return COCO(coco)


def generate_captions(model, images, vocab, max_seq_length, device):
    """
    Generate captions for a batch of images.

    Args:
        model (nn.Module): Trained image captioning model.
        images (torch.Tensor): Batch of images.
        vocab (Vocabulary): Vocabulary object.
        max_seq_length (int): Maximum sequence length for captions.
        device (torch.device): Device to run inference on.

    Returns:
        list: List of generated captions.
    """
    model.eval()
    captions = []

    with torch.no_grad():
        for image in images:
            image = image.unsqueeze(0).to(device)
            caption = generate_caption(model, image, vocab, max_seq_length, device)
            captions.append(caption)

    return captions


def generate_caption(model, image, vocab, max_seq_length, device):
    """
    Generate a caption for a single image.

    Args:
        model (nn.Module): Trained image captioning model.
        image (torch.Tensor): Input image tensor.
        vocab (Vocabulary): Vocabulary object.
        max_seq_length (int): Maximum sequence length for captions.
        device (torch.device): Device to run inference on.

    Returns:
        str: Generated caption.
    """
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