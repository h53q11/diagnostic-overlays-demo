
import numpy as np
import torch

    
def gradient_norm(model):
    """Compute L2 norm of all gradients."""
    total = 0.0
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                total += (p.grad.detach() ** 2).sum().item()
    return np.sqrt(total)


def activation_variance(activations):
    """Simple diagnostic: variance of hidden activations."""
    return activations.var().item()


def gru_activation_variance(hidden_seq):
    """Variance across all hidden states."""
    return hidden_seq.var().item()
