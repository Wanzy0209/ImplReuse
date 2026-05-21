import sys
import platform
import torch

print("Python:", sys.version)
print("Executable:", sys.executable)
print("PyTorch:", torch.__version__)
print("OS:", platform.system())
print("platform:", platform.platform())
print("arch:", platform.machine())

# Check CUDA availability as a prerequisite for cuDNN
print("CUDA available?:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)

# Check the specific API
cudnn_available = torch.backends.cudnn.is_available()
print("cuDNN available?:", cudnn_available)

# Verify the return type
assert isinstance(cudnn_available, bool), "torch.backends.cudnn.is_available should return a boolean"