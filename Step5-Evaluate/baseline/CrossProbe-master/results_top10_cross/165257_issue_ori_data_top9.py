```python
import tensorflow as tf

# Conversion: PyTorch in-place operations are translated to TensorFlow functional operations.
# TensorFlow tensors are immutable, so operations return new tensors rather than modifying in place.
ops = [
    ("normal_(0,1)", lambda t: tf.random.normal(t.shape, mean=0.0, stddev=1.0)),
    ("uniform_(0,1)", lambda t: tf.random.uniform(t.shape, minval=0.0, maxval=1.0)),
    ("exponential_(1)", lambda t: tf.random.gamma(t.shape, alpha=1.0)), # Exponential with rate 1 is Gamma(1, 1)
    ("bernoulli_(0.5)", lambda t: tf.cast(tf.random.uniform(t.shape) < 0.5, tf.float32)),
    ("random_()", lambda t: tf.random.uniform(t.shape)),
    ("random_(10)", lambda t: tf.random.uniform(t.shape, maxval=10)),
    ("random_(0,10)", lambda t: tf.random.uniform(t.shape, minval=0, maxval=10)),
]

print(f"{'Operation':<20} {'CPU Max':<12} {'GPU Max':<12} {'Status'}")
print("-" * 60)

for name, op_func in ops:
    # CPU: works correctly
    # Conversion: Explicitly place operations on CPU
    with tf.device('/cpu:0'):
        # Conversion: torch.zeros(...).T.clone() -> tf.transpose(tf.zeros(...))
        # Note: In TF, .clone() is implicit as operations create new tensors.
        # The transpose creates a tensor with non-standard strides (non-contiguous in PyTorch terms).
        t_cpu = tf.transpose(tf.zeros((50, 50)))
        t_cpu = op_func(t_cpu)
        cpu_max = tf.reduce_max(t_cpu).numpy()

    # MPS: Test on accelerator
    # Conversion: PyTorch 'mps' device maps to TensorFlow '/GPU:0' (which uses MPS backend on Mac)
    with tf.device('/GPU:0'):
        t_gpu = tf.transpose(tf.zeros((50, 50)))
        t_gpu = op_func(t_gpu)
        gpu_max = tf.reduce_max(t_gpu).numpy()

    # Conversion: Check if operation produced non-zero values
    # TF functional ops generally do not fail silently like the in-place PyTorch MPS bug
    status = "✓ OK" if gpu_max != 0.0 else "✗ BUG"
    print(f"{name:<20} {cpu_max:<12.4f} {gpu_max:<12.4f} {status}")
```