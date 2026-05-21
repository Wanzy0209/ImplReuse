import torch
import time

def test_matmul_performance_regression():
    """
    Test case to reproduce torch.matmul regression on CPU.
    Issue ID: 162683
    
    This test benchmarks the matrix multiplication operation for specific shapes
    and data types to detect performance regressions between PyTorch versions.
    """
    torch.manual_seed(0)
    
    # Shapes defined in the original bug report
    shapes = [
        ((1, 12, 10, 64), (1, 12, 64, 10)),
        ((1, 12, 10, 10), (1, 12, 10, 64)),
    ]

    device = "cpu"
    dtype = torch.float16

    # Context check pattern inspired by the similar API (global_variables_initializer)
    # to ensure the environment is suitable for the test.
    if device == "cpu":
        # Proceed with CPU execution
        pass
    else:
        print(f"Skipping test: Target device {device} not available or not CPU.")
        return

    for a_shape, b_shape in shapes:
        print(f"Benchmarking shape: {a_shape} x {b_shape}")
        
        # Initialize tensors
        A = torch.empty(a_shape, dtype=dtype, device=device).uniform_(0,1) * 2 - 1
        B = torch.empty(b_shape, dtype=dtype, device=device).uniform_(0,1) * 2 - 1

        # Warm up phase to mitigate cold start effects
        for _ in range(5000):
            _ = torch.matmul(A, B)

        # Benchmarking phase
        repeat = 500
        times = []
        for i in range(repeat):
            start = time.perf_counter()
            _ = torch.matmul(A, B)
            end = time.perf_counter()
            # Discard first 100 iterations to allow for stabilization
            if i > 100:
                times.append((end - start) * 1e6) # Convert to microseconds
        
        times.sort()
        avg_time_us = sum(times) / len(times)
        print(f"Average execution time: {avg_time_us:.3f} us")
        
        # Assertion to verify correctness of the operation
        C = torch.matmul(A, B)
        expected_shape = torch.Size([a_shape[0], a_shape[1], a_shape[2], b_shape[3]])
        assert C.shape == expected_shape, f"Output shape mismatch: {C.shape} vs {expected_shape}"

if __name__ == "__main__":
    test_matmul_performance_regression()