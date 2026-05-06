uv pip install torch

python -c 'import torch; print(torch.__version__); import torch.ops.symm_mem'

2.9.0+cu128
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'torch.ops.symm_mem'