```python
import tensorflow as tf

# Conversion: PyTorch in-place ops are translated to TensorFlow functional ops.
# TensorFlow tensors are immutable, so operations return new tensors.
ops = [
    ("normal_(0,1)", lambda t: tf.random.normal(shape=t.shape, mean=0.0, stddev=1.0)),
    ("uniform_(0,1)", lambda t: tf.random.uniform(shape=t.shape, minval=0, maxval=1)),
    ("exponential_(1)", lambda t: tf.random.gamma(shape=t.shape, alpha=1.0, beta=1.0)), # Exponential(1) is Gamma(1, 1)
    ("bernoulli_(0.5)", lambda t: tf.cast(tf.random.uniform(shape=t.shape) < 0.5, tf.float32)),
    ("random_()", lambda t: tf.random.uniform(shape=t.shape, minval=0, maxval=1)),
    ("random_(10)", lambda t: tf.random.uniform(shape=t.shape, minval=0, maxval=10, dtype=tf.int32)),
    ("random_(0,10)", lambda t: tf.random.uniform(shape=t.shape, minval=0, maxval=10, dtype=tf.int32)),
]

print(f"{'Operation':<20} {'CPU Max':<12} {'GPU Max':<12} {'Status'}")
print("-" * 60)

# Check for GPU availability to replace MPS
gpus = tf.config.list_physical_devices('GPU')
has_gpu = len(gpus) > 0

for name, op_func in ops:
    # CPU: works correctly
    with tf.device('/CPU:0'):
        # Conversion: Mimic non-contiguous tensor creation (zeros -> transpose)
        t_cpu = tf.transpose(tf.zeros((50, 50)))
        t_cpu = op_func(t_cpu)
        cpu_max = tf.reduce_max(t_cpu).numpy()

    # GPU: mapped from MPS (Metal Performance Shaders)
    if has_gpu:
        with tf.device('/GPU:0'):
            t_gpu = tf.transpose(tf.zeros((50, 50)))
            t_gpu = op_func(t_gpu)
            gpu_max = tf.reduce_max(t_gpu).numpy()
    else:
        # Fallback if no GPU is available
        gpu_max = 0.0

    status = "✓ OK" if gpu_max != 0.0 else "✗ BUG"
    print(f"{name:<20} {cpu_max:<12.4f} {gpu_max:<12.4f} {status}")
```