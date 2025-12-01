# import standard libraries
import sys

# import 3rd party libraries
import numpy as np
import torch
from pathlib import Path

# import local libraries
from PyPruning.GreedyPruningClassifier import *

# --- Example Usage ---
# Load data
try:
    proba = Path(sys.argv[1]).resolve()
    target = Path(sys.argv[2]).resolve()

    assert proba.is_file(), f"Proba file does not exist or is not a file: {proba}"
    assert target.is_file(), f"Target file does not exist or is not a file: {target}"

except Exception as e:
        sys.exit(f'\nAn error occured: {e}.\n\nUsage: python test_predictions_PyPruning.py /path/to/predictions.pt /path/to/targets.pt')

proba = torch.load(proba).float().numpy()
target = torch.load(target).float().numpy()

# Prune
print("\n--- Output Shapes ---")
print(f"Proba matrix shape (M, N, C): {proba.shape}")
print(f"Target array shape (N,):    {target.shape}")

pruner = GreedyPruningClassifier(n_estimators=2, metric=reduced_error)
selected_indices, _ = pruner.prune_(proba, target)

print(f"\nSelected model indices: {selected_indices}")
