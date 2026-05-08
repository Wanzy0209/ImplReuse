!pip uninstall -y torch

!pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu129

import torch
print(torch.__version__)
input = [[()],{},[torch.empty((9, 6, 3, 6, 9), dtype=torch.complex128, device='cuda'),torch.empty((5, 7, 9, 8, 5), dtype=torch.uint32, device='cuda')],{}]
r1 = torch.nn.MaxUnpool3d(*input[0],**input[1])
r2 = r1(*input[2],**input[3])