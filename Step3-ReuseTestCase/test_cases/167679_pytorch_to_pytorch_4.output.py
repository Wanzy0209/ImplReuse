import sys
import platform
import torch
import torch.distributed

print("Python:", sys.version)
print("Executable:", sys.executable)
print("PyTorch:", torch.__version__)
print("OS:", platform.platform())
print("Arch:", platform.machine())

# Check general distributed availability
print("Distributed available?:", torch.distributed.is_available())

# Check specific MPI backend availability
print("MPI available?:", torch.distributed.is_mpi_available())

# Verify the return type is boolean
assert isinstance(torch.distributed.is_mpi_available(), bool), "is_mpi_available should return a boolean"