import torch.nn as nn


def get_loss_function(vocab):
    """
    Returns the loss function for the model.

    Args:
        vocab (Vocabulary): Vocabulary object containing padding token index.

    Returns:
        nn.CrossEntropyLoss: Loss function with padding index ignored.
    """
    return nn.CrossEntropyLoss(ignore_index=vocab("<pad>"))
