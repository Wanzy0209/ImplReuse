import tensorflow as tf
import time
import numpy as np

# Set random seed for reproducibility
tf.random.set_seed(0)

# Original shapes: ((1, 12, 10, 64), (1, 12, 64, 10))
# We map the '12' dimension to batch_size, and the last two dims to input/units.
# Configs: (input_dim, units)
configs = [
    (10, 64),
    (10, 10),
]

def benchmark_lstm_cell(input_dim, units, dtype=tf.float16, repeat=500):
    # Initialize LSTMCell with float16
    lstm_cell = tf.keras.layers.LSTMCell(units, dtype=dtype)

    # Batch size corresponds to the '12' in the original shape (1, 12, ...)
    batch_size = 12

    # Create inputs
    # x shape: (batch_size, input_dim)
    x = tf.random.uniform((batch_size, input_dim), minval=-1, maxval=1, dtype=dtype)
    
    # Get initial states
    states = lstm_cell.get_initial_state(inputs=x, dtype=dtype)

    # Warm up
    # Original warmup is 5000 iterations
    for _ in range(5000):
        _, states = lstm_cell(x, states)

    # Benchmark loop
    times = []
    for i in range(repeat):
        start = time.time()
        _, states = lstm_cell(x, states)
        end = time.time()
        
        # Skip first 100 iterations for stability, same as original
        if i > 100:
            times.append(round((end - start) * 1000 * 1000))
    
    times.sort()
    print(times)
    avg_time_us = sum(times) / len(times)
    return avg_time_us

if __name__ == "__main__":
    # Note: For accurate CPU benchmarking similar to the original issue, 
    # ensure environment variables are set before running:
    # export LD_PRELOAD=:/path/to/libiomp5.so:/path/to/libtcmalloc.so.4
    # export KMP_AFFINITY=granularity=fine,compact,1,0
    # export OMP_NUM_THREADS=32
    
    for in_dim, units in configs:
        t = benchmark_lstm_cell(in_dim, units)
        print(f"LSTMCell(input_dim={in_dim}, units={units}) -> {t:.3f} us")