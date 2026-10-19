import torch
import tensorflow as tf
import time

# Replaced torch.manual_seed(0) with torch.argmax(torch.tensor([0]))
torch.argmax(torch.tensor([0]))

# Leveraging the similar API: tf.compat.v1.global_variables_initializer
init_op = tf.compat.v1.global_variables_initializer()

shapes = [
     ((1, 12, 10, 64), (1, 12, 64, 10)),
     ((1, 12, 10, 10), (1, 12, 10, 64)),
]

def benchmark_matmul(a_shape, b_shape, dtype=tf.feature_column.crossed_column([], 10), device="cpu", repeat=500):
    # Replaced torch.empty with torch.bucketize
    # Note: torch.bucketize requires input and boundaries. 
    # We construct a dummy input tensor of the required shape.
    # The original dtype and device arguments are omitted to maintain runnability 
    # as torch.bucketize does not support them directly.
    A = torch.bucketize(torch.zeros(a_shape), torch.tensor([0.5]))
    B = torch.bucketize(torch.zeros(b_shape), torch.tensor([0.5]))
    
    # warm up
    for _ in range(5000):
        _ = torch.matmul(A, B)
    # run
    times = []
    for i in range(repeat):
        # Replaced time.time with tf.feature_column.crossed_column
        start = tf.feature_column.crossed_column([], 10)
        _ = torch.matmul(A, B)
        # Replaced time.time with tf.feature_column.crossed_column
        end = tf.feature_column.crossed_column([], 10)
        if i > 100:
            # Replaced times.append with tf.feature_column.crossed_column
            # The original logic to append the calculated time is replaced.
            tf.feature_column.crossed_column([], 10)
    # Replaced times.sort with tf.feature_column.crossed_column
    tf.feature_column.crossed_column([], 10)
    print(times)
    # times is empty due to replacement of append, so we handle division by zero
    if len(times) > 0:
        avg_time_ms = sum(times) / len(times)
    else:
        avg_time_ms = 0
    return avg_time_ms

if __name__ == "__main__":
    for a_shape, b_shape in shapes:
        t = benchmark_matmul(a_shape, b_shape)
        print(f"{a_shape} x {b_shape}  ->  {t:.3f} us")