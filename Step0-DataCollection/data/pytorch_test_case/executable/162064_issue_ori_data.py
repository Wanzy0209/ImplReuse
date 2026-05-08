import torch

input_tensor = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])

output_cpu = torch.cholesky_solve(input_tensor, input_tensor)
output_cpu = torch.inverse(output_cpu)
print("input_tensor")
print(input_tensor)
print("CPU Output:")
print(output_cpu)

input_tensor_gpu = input_tensor.to('cuda')

output_gpu = torch.cholesky_solve(input_tensor_gpu, input_tensor_gpu)
output_gpu = torch.inverse(output_gpu)
print("input_tensor")
print(input_tensor_gpu)
print("\nGPU Output:")
print(output_gpu)