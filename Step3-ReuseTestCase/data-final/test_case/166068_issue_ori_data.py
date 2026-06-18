import torch
from packaging import version

print(version.parse(torch.version.hip))