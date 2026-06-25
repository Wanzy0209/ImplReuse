import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'GPU Count: {torch.cuda.device_count()}')