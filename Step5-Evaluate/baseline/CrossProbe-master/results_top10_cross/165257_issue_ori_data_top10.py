```python
import tensorflow as tf

# Test each operation on CPU and GPU
# Note: TensorFlow tensors are immutable, so operations return new tensors
# rather than modifying in-place. We simulate the logic by applying the
# operation to the transposed (non-contiguous) tensor.
ops = [
    ("normal_(0,1)", lambda t: tf.random.normal(tf.shape(t), 0, 1)),
    ("uniform_(0,1)", lambda t: tf.random.uniform(tf.shape(t), 0, 1)),
    ("exponential_(1)", lambda t: tf.random.gamma(tf.shape(t), 1.0, 1.0)), # Gamma(1, 1) is Exponential(1)
    ("bernoulli_(0.5)", lambda t: tf.cast(tf.random.uniform(tf.shape(t)) < 0.5, t.dtype)),
    ("random_()", lambda t: tf.random.uniform(tf.shape(t))),
    ("random_(10)", lambda t: tf.cast(tf.random.uniform(tf.shape(t), maxval=10), t.dtype)),
    ("random_(0,10)", lambda t: tf.cast(tf.random.uniform(tf.shape(t), minval=0, maxval=10), t.dtype)),
]

print(f"{'Operation':<20} {'CPU Max':<12} {'GPU Max':<12} {'Status'}")
print("-" * 60)

for name, op_func in ops:
    # CPU: works correctly
    with tf.device('/CPU:0'):
        # Create non-contiguous tensor (transpose)
        t_cpu = tf.transpose(tf.zeros((50, 50)))
        # Apply operation (returns new tensor)
        t_cpu = op_func(t_cpu)
        cpu_max = tf.reduce_max(t_cpu).numpy()

    # GPU: TensorFlow handles non-contiguous tensors automatically
    # Mapping 'mps' to '/GPU:0' (standard TensorFlow device placement)
    with tf.device('/GPU:0'):
        # Create non-contiguous tensor (transpose)
        t_gpu = tf.transpose(tf.zeros((50, 50)))
        # Apply operation (returns new tensor)
        t_gpu = op_func(t_gpu)
        gpu_max = tf.reduce_max(t_gpu).numpy()

    status = "✓ OK" if gpu_max != 0.0 else "✗ BUG"
    print(f"{name:<20} {cpu_max:<12.4f} {gpu_max:<12.4f} {status}")
```