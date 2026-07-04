# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")  # Returns False
print(f"GPU Count: {torch.cuda.device_count()}")       # Returns 0