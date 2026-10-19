```python
import tensorflow as tf

# Conversion: PyTorch in-place random ops are replaced by TensorFlow functional ops.
# TensorFlow tensors are immutable, so we reassign the result.
# PyTorch .T.clone() (non-contiguous) is approximated by tf.transpose.
ops = [
    ("normal_(0,1)", lambda t: tf.random.normal(shape=t.shape, mean=0.0, stddev=1.0)),
    ("uniform_(0,1)", lambda t: tf.random.uniform(shape=t.shape, minval=0.0, maxval=1.0)),
    ("exponential_(1)", lambda t: tf.random.exponential(shape=t.shape, lam=1.0)),
    ("bernoulli_(0.5)", lambda t: tf.cast(tf.random.uniform(shape=t.shape) < 0.5, dtype=t.dtype)),
    ("random_()", lambda t: tf.random.uniform(shape=t.shape, minval=0.0, maxval=1.0)),
    ("random_(10)", lambda t: tf.random.uniform(shape=t.shape, minval=0, maxval=10, dtype=tf.int64)),
    ("random_(0,10)", lambda t: tf.random.uniform(shape=t.shape, minval=0, maxval=10, dtype=tf.int64)),
]

print(f"{'Operation':<20} {'CPU Max':<12} {'GPU Max':<12} {'Status'}")
print("-" * 60)

# Check for GPU availability to map MPS to GPU
gpus = tf.config.list_physical_devices('GPU')
device_name = '/GPU:0' if gpus else '/CPU:0'

for name, op_func in ops:
    # CPU: works correctly
    with tf.device('/CPU:0'):
        t_cpu = tf.zeros((50, 50))
        t_cpu = tf.transpose(t_cpu)  # Equivalent to .T.clone() (non-contiguous)
        t_cpu = op_func(t_cpu)
        cpu_max = tf.reduce_max(t_cpu).numpy()

    # GPU: Test on GPU (equivalent to MPS in PyTorch)
    with tf.device(device_name):
        t_gpu = tf.zeros((50, 50))
        t_gpu = tf.transpose(t_gpu)  # Equivalent to .T.clone()
        t_gpu = op_func(t_gpu)
        gpu_max = tf.reduce_max(t_gpu).numpy()

    status = "✓ OK" if gpu_max != 0.0 else "✗ BUG"
    print(f"{name:<20} {cpu_max:<12.4f} {gpu_max:<12.4f} {status}")
```