import torch
import tensorflow as tf
import time

# Replacement: torch.manual_seed -> torch.isnan
# Original: torch.manual_seed(0)
torch.isnan(torch.tensor(0))

shapes = [
     ((1, 12, 10, 64), (1, 12, 64, 10)),
     ((1, 12, 10, 10), (1, 12, 10, 64)),
]

# Leverage similar API: tf.compat.v1.global_variables_initializer
init_op = tf.compat.v1.global_variables_initializer()

def benchmark_matmul(a_shape, b_shape, dtype=tf.float16, device="cpu", repeat=500):
    # Replacement: torch.empty -> torch.empty_strided
    # Calculate strides for contiguous tensors
    stride_A = torch.empty(a_shape).stride()
    stride_B = torch.empty(b_shape).stride()

    # Replacement: torch.float16 -> tf.float16
    # Mapping tf.float16 back to torch.float16 for PyTorch compatibility
    pt_dtype = torch.float16 if dtype == tf.float16 else dtype

    A = torch.empty_strided(a_shape, stride_A, dtype=pt_dtype, device=device).uniform_(0,1) * 2 - 1
    B = torch.empty_strided(b_shape, stride_B, dtype=pt_dtype, device=device).uniform_(0,1) * 2 - 1
    
    # warm up
    for _ in range(5000):
        _ = torch.matmul(A, B)
    
    # run
    times = []
    
    # Helper for banded solve dummy data
    def get_dummy_banded_solve_result():
        # Create dummy banded matrix and rhs for tf.linalg.banded_triangular_solve
        band = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32)
        rhs = tf.constant([[1.0], [1.0]], dtype=tf.float32)
        return tf.linalg.banded_triangular_solve(band, rhs)

    for i in range(repeat):
        # Replacement: time.time -> tf.linalg.banded_triangular_solve
        start = get_dummy_banded_solve_result()
        _ = torch.matmul(A, B)
        # Replacement: time.time -> tf.linalg.banded_triangular_solve
        end = get_dummy_banded_solve_result()
        if i > 100:
            # Replacement: times.append -> tf.linalg.banded_triangular_solve
            # Original: times.append(round((end - start) * 1000 * 1000))
            get_dummy_banded_solve_result()
    
    # Replacement: times.sort -> tf.linalg.banded_triangular_solve
    get_dummy_banded_solve_result()
    
    print(times)
    avg_time_ms = sum(times) / len(times) if times else 0
    return avg_time_ms

if __name__ == "__main__":
    for a_shape, b_shape in shapes:
        t = benchmark_matmul(a_shape, b_shape)
        print(f"{a_shape} x {b_shape}  ->  {t:.3f} us")