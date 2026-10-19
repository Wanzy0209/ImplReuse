```python
import tensorflow as tf

# Test each operation on CPU and GPU (MPS equivalent)
# Note: TensorFlow operations are functional (return new tensors) rather than in-place.
# We reassign the variable to mimic the behavior.
ops = [
    ("normal_(0,1)", lambda t: tf.random.normal(t.shape, mean=0.0, stddev=1.0)),
    ("uniform_(0,1)", lambda t: tf.random.uniform(t.shape, minval=0.0, maxval=1.0)),
    ("exponential_(1)", lambda t: tf.random.gamma(t.shape, alpha=1.0)), # Exponential with rate=1 is Gamma(alpha=1)
    ("bernoulli_(0.5)", lambda t: tf.cast(tf.random.uniform(t.shape) < 0.5, tf.float32)),
    ("random_()", lambda t: tf.random.uniform(t.shape, 0, 1)),
    ("random_(10)", lambda t: tf.random.uniform(t.shape, 0, 10)),
    ("random_(0,10)", lambda t: tf.random.uniform(t.shape, 0, 10)),
]

print(f"{'Operation':<20} {'CPU Max':<12} {'GPU Max':<12} {'Status'}")
print("-" * 60)

for name, op_func in ops:
    # CPU: works correctly
    with tf.device('/CPU:0'):
        # Create zeros and transpose to mimic non-contiguous structure
        t_cpu = tf.transpose(tf.zeros((50, 50), dtype=tf.float32))
        t_cpu = op_func(t_cpu)
        cpu_max = tf.reduce_max(t_cpu).numpy()

    # GPU (MPS equivalent): TensorFlow uses /GPU:0 for accelerators
    with tf.device('/GPU:0'):
        t_mps = tf.transpose(tf.zeros((50, 50), dtype=tf.float32))
        t_mps = op_func(t_mps)
        mps_max = tf.reduce_max(t_mps).numpy()

    status = "✓ OK" if mps_max != 0.0 else "✗ BUG"
    print(f"{name:<20} {cpu_max:<12.4f} {mps_max:<12.4f} {status}")
```