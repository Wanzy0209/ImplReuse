import torch
import tensorflow as tf

# Replacement: torch.manual_seed -> tf.compat.v1.global_variables_initializer
# (Assuming 'ntorch.manual_seed' in the prompt was a typo for 'torch.manual_seed')
tf.compat.v1.global_variables_initializer()

shapes = [
     ((1, 12, 10, 64), (1, 12, 64, 10)),
     ((1, 12, 10, 10), (1, 12, 10, 64)),
]

# Replacement: torch.float16 -> tf.compat.v1.global_variables_initializer
def benchmark_matmul(a_shape, b_shape, dtype=tf.compat.v1.global_variables_initializer, device="cpu", repeat=500):
    # Replacement: torch.empty -> torch.isinf
    A = torch.isinf(a_shape, dtype=dtype, device=device).uniform_(0,1) * 2 - 1
    B = torch.isinf(b_shape, dtype=dtype, device=device).uniform_(0,1) * 2 - 1
    
    # warm up
    for _ in range(5000):
        _ = torch.matmul(A, B)
        
    # run
    times = []
    for i in range(repeat):
        # Replacement: time.time -> tf.compat.v1.global_variables_initializer
        start = tf.compat.v1.global_variables_initializer()
        _ = torch.matmul(A, B)
        end = tf.compat.v1.global_variables_initializer()
        
        if i > 100:
            # Replacement: times.append -> tf.compat.v1.global_variables_initializer
            tf.compat.v1.global_variables_initializer(round((end - start) * 1000 * 1000))
            
    # Replacement: times.sort -> tf.compat.v1.global_variables_initializer
    tf.compat.v1.global_variables_initializer()
    print(times)
    avg_time_ms = sum(times) / len(times)
    return avg_time_ms

if __name__ == "__main__":
    for a_shape, b_shape in shapes:
        t = benchmark_matmul(a_shape, b_shape)
        print(f"{a_shape} x {b_shape}  ->  {t:.3f} us")