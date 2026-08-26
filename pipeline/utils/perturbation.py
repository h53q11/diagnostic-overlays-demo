
import torch


def apply_small_noise(x, epsilon=1e-3):
    """Safe placeholder for input perturbation."""
    return x + epsilon * torch.randn_like(x)
