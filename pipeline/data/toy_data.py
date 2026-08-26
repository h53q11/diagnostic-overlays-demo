
import numpy as np
from sklearn.datasets import make_regression


# ---------------------------------------------------------
# 1. Toy dataset (tabular)
# ---------------------------------------------------------
def make_toy_data(n_samples=2000, n_features=10, noise=5.0, random_state=0):
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state,
    )
    return X.astype(np.float32), y.astype(np.float32).reshape(-1, 1)

# ---------------------------------------------------------
# 2. Toy sequence dataset
# ---------------------------------------------------------
def make_sequence_data(n_samples=2000, seq_len=15, n_features=10):
    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    y = X.mean(axis=(1, 2)).reshape(-1, 1).astype(np.float32)  # trivial target
    return X, y
