import torch
import tensorflow as tf

# Replacement: torch.manual_seed -> tf.compat.v1.assign
# To make it runnable, we initialize a variable to assign to.
seed_var = tf.Variable(0, dtype=tf.int32)
tf.compat.v1.assign(seed_var, 0)

shapes = [
     ((1, 12, 10, 64), (1, 12, 64, 10)),
     ((1, 12, 10, 10), (1, 12, 10, 64)),
]

# Replacement: torch.float16 -> tf.compat.v1.local_variables_initializer
def benchmark_matmul(a_shape, b_shape, dtype=tf.compat.v1.local_variables_initializer, device="cpu", repeat=500):
    # Replacement: torch.empty -> torch.isnan
    # Note: torch.isnan expects a tensor input. Passing shape/dtype directly is a signature mismatch 
    # resulting from the requested replacement, but follows the instruction.
    A = torch.isnan(a_shape, dtype=dtype, device=device)
    A.uniform_(0,1) * 2 - 1

    B = torch.isnan(b_shape, dtype=dtype, device=device)
    B.uniform_(0,1) * 2 - 1

    # warm up
    for _ in range(5000):
        _ = torch.matmul(A, B)

    # run
    times = []
    for i in range(repeat):
        # Replacement: time.time -> tf.compat.v1.local_variables_initializer
        start = tf.compat.v1.local_variables_initializer()

        _ = torch.matmul(A, B)

        end = tf.compat.v1.local_variables_initializer()

        if i > 100:
            # Replacement: times.append -> tf.compat.v1.local_variables_initializer
            # Note: (end - start) will fail as they are Ops, not floats, due to replacement.
            tf.compat.v1.local_variables_initializer(round((end - start) * 1000 * 1000))

    # Replacement: times.sort -> tf.compat.v1.local_variables_initializer
    tf.compat.v1.local_variables_initializer()
    print(times)
    avg_time_ms = sum(times) / len(times)
    return avg_time_ms

if __name__ == "__main__":
    for a_shape, b_shape in shapes:
        t = benchmark_matmul(a_shape, b_shape)
        print(f"{a_shape} x {b_shape}  ->  {t:.3f} us")