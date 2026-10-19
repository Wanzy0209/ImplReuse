```python
import tensorflow as tf

# Test each operation on CPU and GPU (MPS on Mac)
# Conversion: PyTorch in-place operations (_) are simulated using tf.Variable.assign
ops = [
    ("normal_(0,1)", lambda v: v.assign(tf.random.normal(v.shape, mean=0, stddev=1))),
    ("uniform_(0,1)", lambda v: v.assign(tf.random.uniform(v.shape, minval=0, maxval=1))),
    ("exponential_(1)", lambda v: v.assign(tf.random.gamma(v.shape, alpha=1.0, beta=1.0))), # PyTorch exponential with rate=1 maps to TF Gamma(alpha=1)
    ("bernoulli_(0.5)", lambda v: v.assign(tf.cast(tf.random.uniform(v.shape) < 0.5, tf.float32))),
    ("random_()", lambda v: v.assign(tf.random.uniform(v.shape))),
    ("random_(10)", lambda v: v.assign(tf.random.uniform(v.shape, maxval=10))),
    ("random_(0,10)", lambda v: v.assign(tf.random.uniform(v.shape, minval=0, maxval=10))),
]

print(f"{'Operation':<20} {'CPU Max':<12} {'GPU Max':<12} {'Status'}")
print("-" * 60)

for name, op_func in ops:
    # CPU: works correctly
    # Conversion: torch.zeros -> tf.zeros, .T -> tf.transpose, .clone() -> implicit in Variable creation
    with tf.device('/CPU:0'):
        t_cpu = tf.Variable(tf.transpose(tf.zeros((50, 50))))  # Non-contiguous
    op_func(t_cpu)
    cpu_max = t_cpu.numpy().max()

    # GPU (MPS): fails silently
    # Conversion: device='mps' -> device='/GPU:0' (TF maps GPU to MPS on Mac automatically)
    with tf.device('/GPU:0'):
        t_gpu = tf.Variable(tf.transpose(tf.zeros((50, 50))))  # Non-contiguous
    op_func(t_gpu)
    gpu_max = t_gpu.numpy().max()

    status = "✓ OK" if gpu_max != 0.0 else "✗ BUG"
    print(f"{name:<20} {cpu_max:<12.4f} {gpu_max:<12.4f} {status}")
```