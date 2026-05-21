import sys
import platform
import torch
import torch.distributed

print("Python:", sys.version)
print("Executable:", sys.executable)
print("PyTorch:", torch.__version__)
print("Platform:", platform.platform())
print("Arch:", platform.machine())

# Adapted checks for torch.distributed
print("Distributed available?:", torch.distributed.is_available())
print("Distributed initialized?:", torch.distributed.is_initialized())

# Assertion to verify the API returns a boolean
assert isinstance(torch.distributed.is_initialized(), bool), "is_initialized should return a boolean"