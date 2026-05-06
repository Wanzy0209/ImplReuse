import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "roundup_power2_divisions:[>:1]"

import torch

MB = 1024 * 1024
t = torch.empty([514*MB], dtype=torch.int8, device='cuda')

print(torch.cuda.memory_summary())