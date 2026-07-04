import torch
import time
import torch.nn.functional as F

torch.manual_seed(0)

# Adapted shapes: local_response_norm takes a single input tensor.
# The original shapes were 4D, which is compatible with LRN (N, C, H, W).
shapes = [
    (1, 12, 10, 64),
    (1, 12, 10, 10),
]

# Fix: Changed default dtype from torch.float16 to torch.float32.
# The error "avg_pool3d_out_frame not implemented for 'Half'" indicates that
# float16 is not supported for this operation on CPU.
def benchmark_lrn(input_shape, size, dtype=torch.float32, device="cpu", repeat=500):
    # Create input tensor
    A = torch.empty(input_shape, dtype=dtype, device=device).uniform_(0,1) * 2 - 1
    
    # Warm up
    for _ in range(5000):
        _ = F.local_response_norm(A, size)
    
    # Run
    times = []
    for i in range(repeat):
        start = time.time()
        _ = F.local_response_norm(A, size)
        end = time.time()
        if i > 100:
            times.append(round((end - start) * 1000 * 1000))
    times.sort()
    print(times)
    avg_time_ms = sum(times) / len(times)
    return avg_time_ms

if __name__ == "__main__":
    # Choose a reasonable 'size' parameter for LRN (number of adjacent channels to normalize)
    size = 5
    for shape in shapes:
        t = benchmark_lrn(shape, size)
        print(f"LRN({shape}, size={size}) -> {t:.3f} us")