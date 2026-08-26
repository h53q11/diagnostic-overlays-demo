import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. Simple MLP model
# ---------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        out = self.fc3(h2)
        return out, h2  # return activations for overlay




# ---------------------------------------------------------
# 2. Simple GRU model
# ---------------------------------------------------------
class SimpleGRU(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, output_dim=1, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.gru(x)
        last = out[:, -1, :]  # final timestep
        return self.fc(last), out  # return output + hidden sequence



# ---------------------------------------------------------
# 3. Demonstrator: model initialisation (safe version)
# ---------------------------------------------------------
class DemoMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64), output_dim=1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

