import sys
import platform
import torch
import torch.distributed

print("Python:", sys.version)
print("Executable:", sys.executable)
print("PyTorch:", torch.__version__)
print("OS:", platform.platform())
print("Arch:", platform.machine())

# Check if the distributed package is available at all
print("Distributed available?:", torch.distributed.is_available())

# Check the specific Gloo backend availability
print("Gloo available?:", torch.distributed.is_gloo_available())