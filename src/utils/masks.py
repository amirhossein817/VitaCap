# src/utils/masks.py

import torch


def generate_caption_masks(seq_len):
    """
    Generate target masks for the captions.

    Args:
        seq_len (int): Length of the target sequence.

    Returns:
        torch.Tensor: Target mask of shape [seq_len, seq_len].
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask
