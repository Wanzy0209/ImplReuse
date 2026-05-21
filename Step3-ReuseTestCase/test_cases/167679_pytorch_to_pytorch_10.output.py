import sys
import platform
import torch

print("Python:", sys.version)
print("Executable:", sys.executable)
print("PyTorch:", torch.__version__)
print("Platform:", platform.platform())
print("Arch:", platform.machine())

# Adapted checks for torch.backends.mkl
# MKL (Math Kernel Library) is typically available on x86_64 builds
print("MKL available?:", torch.backends.mkl.is_available())

# Check if MKL is built, handling cases where is_built might not be exposed
print("MKL built?:", getattr(torch.backends.mkl, "is_built", lambda: "N/A")())