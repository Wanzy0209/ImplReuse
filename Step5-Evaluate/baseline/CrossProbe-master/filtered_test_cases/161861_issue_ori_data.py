import torch
from torch import _inductor
# Reproduce by compiling chroma workflow on SM_75 GPU
# Requires ComfyUI setup and chroma model loading
# Expected: static output on RTX 3090 (SM_75), normal on Ampere