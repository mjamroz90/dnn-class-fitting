"""
Module contains evaluation metrics for experiments in PyTorch
"""
import torch


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor):
    """

    Args:
        y_true: targets
        y_pred: predictions

    Returns:
        Accuracy as scalar
    """
    correct = (y_pred == y_true).sum().item()
    return correct / y_true.size(0)
