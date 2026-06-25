import os
if 'TORCH_CUDA_MEMORY_FRACTION' in os.environ:
    torch.cuda.set_per_process_memory_fraction(float(os.environ['TORCH_CUDA_MEMORY_FRACTION']))