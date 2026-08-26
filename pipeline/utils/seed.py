
import numpy as np
import torch

# ---------------------------------------------------------
# 0. Reproducibility helpers
# ---------------------------------------------------------
def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

