# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
from torch.nn.parallel import comm

tensor1 = torch.rand(2, device=torch.device("cuda:0"))
tensor2 = torch.rand(2, device=torch.device("cuda:1"))
print(f"tensor1: {tensor1}, tensor2: {tensor2}")
print(f'combined tensor - Cuda: {comm.gather([tensor1, tensor2], destination = torch.device("cuda:0"))}')
print(f'combined tensor - CPU: {comm.gather([tensor1, tensor2], destination = torch.device("cpu"))}')