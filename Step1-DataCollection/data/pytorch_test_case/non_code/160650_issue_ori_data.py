# Install mypy
pip install mypy mypy-extensions

# Run mypy from PyTorch root directory
mypy

# Output:
torch/utils/_cxx_pytree.py:1014:1: error: Cannot inherit from final class "PyTreeSpec"  [misc]
Found 1 error in 1 file (checked 1935 source files)