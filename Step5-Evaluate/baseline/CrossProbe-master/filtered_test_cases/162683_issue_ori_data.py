import torch
import time

torch.manual_seed(0)
shapes = [((1, 12, 10, 64), (1, 12, 64, 10)), ((1, 12, 10, 10), (1, 12, 10, 64))]

def benchmark_matmul(a_shape, b_shape, dtype=torch.float16, device="cpu", repeat=500):
    A = torch.empty(a_shape, dtype=dtype, device=device).uniform_(0,1) * 2 - 1
    B = torch.empty(b_shape, dtype=dtype, device=device).uniform_(0,1) * 2 - 1
    for _ in range(5000):
        _ = torch.matmul(A, B)
    times = []
    for i in range(repeat):
        start = time.time()
        _ = torch.matmul(A, B)
        end = time.time()
        if i > 100:
            times.append(round((end - start) * 1000 * 1000))
    times.sort()
    avg_time_ms = sum(times) / len(times)
    return avg_time_ms

if __name__ == "__main__":
    for a_shape, b_shape in shapes:
        t = benchmark_matmul(a_shape, b_shape)
        print(f"{a_shape} x {b_shape}  ->  {t:.3f} us")