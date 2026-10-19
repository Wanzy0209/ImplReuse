```python
import tensorflow as tf

# Conversion: PyTorch in-place ops are simulated in TF using Variable.assign()
# Conversion: PyTorch 'mps' device maps to TensorFlow '/GPU:0'
# Conversion: PyTorch .T.clone() maps to tf.transpose() on a Variable
ops = [
    ("normal_(0,1)", lambda t: t.assign(tf.random.normal(t.shape, mean=0.0, stddev=1.0))),
    ("uniform_(0,1)", lambda t: t.assign(tf.random.uniform(t.shape, minval=0.0, maxval=1.0))),
    ("exponential_(1)", lambda t: t.assign(tf.random.gamma(t.shape, alpha=1.0))), # Exp(1) is Gamma(1, 1)
    ("bernoulli_(0.5)", lambda t: t.assign(tf.cast(tf.random.uniform(t.shape) < 0.5, t.dtype))),
    ("random_()", lambda t: t.assign(tf.random.uniform(t.shape))),
    ("random_(10)", lambda t: t.assign(tf.random.uniform(t.shape, maxval=10.0))),
    ("random_(0,10)", lambda t: t.assign(tf.random.uniform(t.shape, minval=0.0, maxval=10.0))),
]

print(f"{'Operation':<20} {'CPU Max':<12} {'GPU Max':<12} {'Status'}")
print("-" * 60)

# Check for GPU availability to avoid errors if no GPU is present
gpus = tf.config.list_physical_devices('GPU')
has_gpu = len(gpus) > 0

for name, op_func in ops:
    # CPU: works correctly
    with tf.device('/CPU:0'):
        # Conversion: torch.zeros -> tf.Variable(tf.zeros)
        # Conversion: .T.clone() -> assign(tf.transpose(...))
        t_cpu = tf.Variable(tf.zeros((50, 50)))
        t_cpu.assign(tf.transpose(t_cpu))  # Mimic non-contiguous setup
        op_func(t_cpu)
        # Conversion: .max().item() -> tf.reduce_max(...).numpy()
        cpu_max = tf.reduce_max(t_cpu).numpy()

    # GPU: test (mapped from MPS)
    if has_gpu:
        with tf.device('/GPU:0'):
            t_gpu = tf.Variable(tf.zeros((50, 50)))
            t_gpu.assign(tf.transpose(t_gpu))  # Mimic non-contiguous setup
            op_func(t_gpu)
            gpu_max = tf.reduce_max(t_gpu).numpy()
    else:
        # Fallback if no GPU is found
        gpu_max = 0.0

    status = "✓ OK" if gpu_max != 0.0 else "✗ BUG"
    print(f"{name:<20} {cpu_max:<12.4f} {gpu_max:<12.4f} {status}")
```