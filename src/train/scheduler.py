import torch.optim as optim


def get_optimizer(model, learning_rate):
    """
    Returns the optimizer for the model.

    Args:
        model (nn.Module): The model to optimize.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        optim.Adam: Adam optimizer.
    """
    return optim.Adam(model.parameters(), lr=learning_rate)


def get_scheduler(optimizer):
    """
    Returns a learning rate scheduler.

    Args:
        optimizer (optim.Optimizer): Optimizer to apply the scheduler to.

    Returns:
        optim.lr_scheduler.StepLR: Step learning rate scheduler.
    """
    return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
