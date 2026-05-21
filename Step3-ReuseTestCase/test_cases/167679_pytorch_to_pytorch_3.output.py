import sys
import platform
import torch
import torch.distributed.distributed_c10d

print("Python:", sys.version)
print("Executable:", sys.executable)
print("PyTorch:", torch.__version__)
print("Platform:", platform.platform())
print("Arch:", platform.machine())

# Check XCCL availability
is_available = torch.distributed.distributed_c10d.is_xccl_available()
print("XCCL available?:", is_available)

# Assertion to verify the API returns a boolean
assert isinstance(is_available, bool), "is_xccl_available should return a boolean"