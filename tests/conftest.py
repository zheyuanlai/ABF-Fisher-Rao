import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

# Stage-0 tests are numerics tests: run them on CPU for determinism and so the suite
# passes on any node; GPU-specific equivalence tests request cuda explicitly.
DEVICE = torch.device("cpu")
DTYPE = torch.float64
