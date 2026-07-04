```python
import tensorflow as tf

# Conversion: PyTorch in-place random operations are replaced by TensorFlow functional operations.
# Since TF tensors are immutable, the operations return new tensors rather than modifying in-place.
# We cast integer results to float32 to maintain consistent output formatting with the original code.
ops = [
    ("normal_(0,1)", lambda t: tf.random.normal(t.shape, mean=0.0, stddev=1.0)),
    ("uniform_(0,1)", lambda t: tf.random.uniform(t.shape, minval=0.0, maxval=1.0)),
    # PyTorch exponential_(lambda) is equivalent to Gamma(alpha=1, beta=lambda)
    ("exponential_(1)", lambda t: tf.random.gamma(t.shape, alpha=1.0, beta=1.0)),
    ("bernoulli_(0.5)", lambda t: tf.cast(tf.random.uniform(t.shape) < 0.5, dtype=tf.float32)),
    ("random_()", lambda t: tf.random.uniform(t.shape)),
    # PyTorch random_(to) returns integers [0, to). Cast to float for consistent printing.
    ("random_(10)", lambda t: tf.cast(tf.random.uniform(t.shape, maxval=10, dtype=tf.int32), dtype=tf.float32)),
    # PyTorch random_(from, to) returns integers [from, to). Cast to float for consistent printing.
    ("random_(0,10)", lambda t: tf.cast(tf.random.uniform(t.shape, minval=0, maxval=10, dtype=tf.int32), dtype=tf.float32)),
]

print(f"{'Operation':<20} {'CPU Max':<12} {'GPU Max':<12} {'Status'}")
print("-" * 60)

# Check for GPU availability (MPS equivalent on Mac is /GPU:0)
gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
device_name = '/GPU:0' if gpu_available else '/CPU:0'

for name, op_func in ops:
    # CPU: works correctly
    with tf.device('/CPU:0'):
        # Conversion: torch.zeros(...).T.clone() creates a non-contiguous tensor.
        # In TF, we use tf.transpose to change the layout.
        t_cpu = tf.transpose(tf.zeros((50, 50)), perm=[1, 0])
        t_cpu = op_func(t_cpu)
        cpu_max = tf.reduce_max(t_cpu).numpy()

    # GPU (MPS equivalent)
    with tf.device(device_name):
        t_gpu = tf.transpose(tf.zeros((50, 50)), perm=[1, 0])
        t_gpu = op_func(t_gpu)
        gpu_max = tf.reduce_max(t_gpu).numpy()

    status = "✓ OK" if gpu_max != 0.0 else "✗ BUG"
    print(f"{name:<20} {cpu_max:<12.4f} {gpu_max:<12.4f} {status}")
```