# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch

input_tensor = torch.tensor([[1, 2, 3], [3, 2, 1], [1, 1, 1]], dtype=torch.float32)

pinverse_tensor_cpu = torch.pinverse(input_tensor)

print("input_tensor: ")
print(input_tensor)
print("CPU Pseudo-inverse:")
print(pinverse_tensor_cpu)

input_tensor_gpu = input_tensor.to('cuda')

pinverse_tensor_gpu = torch.pinverse(input_tensor_gpu)

print("input_tensor_gpu: ")
print(input_tensor_gpu)
print("\nGPU Pseudo-inverse:")
print(pinverse_tensor_gpu)