
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

from pipeline.models.models import SimpleGRU
from pipeline.data.toy_data import make_sequence_data
from pipeline.diagnostics.basic_metrics import gradient_norm, gru_activation_variance


# ---------------------------------------------------------
#  GRU demo
# ---------------------------------------------------------
def _run_gru_demo(config):
    X, y = make_sequence_data(
        n_samples=config["n_samples"],
        seq_len=config["seq_len"],
        n_features=config["n_features"],
    )
    X_tensor = torch.from_numpy(X).to(device)
    y_tensor = torch.from_numpy(y).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    model = SimpleGRU(input_dim=X.shape[2], hidden_dim=config["gru_hidden_dim"]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    overlay_loss = []
    overlay_actvar = []
    overlay_grad = []

    step = 0
    for epoch in range(config["epochs"]):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds, hidden_seq = model(xb)
            loss = criterion(preds, yb)
            loss.backward()

            # Diagnostics
            gnorm = gradient_norm(model)
            avar = gru_activation_variance(hidden_seq)

            overlay_loss.append((step, loss.item()))
            overlay_grad.append((step, gnorm))
            overlay_actvar.append((step, avar))

            optimizer.step()
            step += 1

        print(f"[GRU] Epoch {epoch+1}: last batch loss = {loss.item():.4f}")

    # Plot
    steps, gvals = zip(*overlay_grad)
    _, avars = zip(*overlay_actvar)
    _, lvals = zip(*overlay_loss)

    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    ax[0].plot(steps, gvals, color="purple")
    ax[0].set_ylabel("Gradient Norm")
    ax[0].set_title("GRU Diagnostic: Gradient Noise")

    ax[1].plot(steps, avars, color="red")
    ax[1].set_ylabel("Activation Variance")
    ax[1].set_title("GRU Diagnostic: Hidden Stability")

    ax[2].plot(steps, lvals, color="black")
    ax[2].set_ylabel("Loss")
    ax[2].set_xlabel("Training Step")
    ax[2].set_title("GRU Training Loss")

    plt.tight_layout()
    plt.show()
    
def run_gru_demo():
    config = {
        "n_samples": 2000,
        "seq_len": 15,
        "n_features": 10,
        "batch_size": 64,
        "epochs": 5,
        "gru_hidden_dim": 32,
        "learning_rate": 1e-3,
    }
    return _run_gru_demo(config)
