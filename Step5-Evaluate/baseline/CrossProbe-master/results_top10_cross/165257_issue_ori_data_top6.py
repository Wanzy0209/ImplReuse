```python
import tensorflow as tf

# Conversion: PyTorch in-place random ops are simulated using tf.Variable.assign
# Conversion: PyTorch MPS device maps to TensorFlow GPU (which uses MPS backend on Mac)
ops = [
    ("normal_(0,1)", lambda v: v.assign(tf.random.normal(v.shape, mean=0.0, stddev=1.0))),
    ("uniform_(0,1)", lambda v: v.assign(tf.random.uniform(v.shape, minval=0.0, maxval=1.0))),
    ("exponential_(1)", lambda v: v.assign(tf.random.gamma(v.shape, alpha=1.0))), # Gamma(1) is equivalent to Exponential(1)
    ("bernoulli_(0.5)", lambda v: v.assign(tf.cast(tf.random.uniform(v.shape) < 0.5, tf.float32))),
    ("random_()", lambda v: v.assign(tf.random.uniform(v.shape))),
    ("random_(10)", lambda v: v.assign(tf.random.uniform(v.shape, maxval=10.0))),
    ("random_(0,10)", lambda v: v.assign(tf.random.uniform(v.shape, minval=0.0, maxval=10.0))),
]

print(f"{'Operation':<20} {'CPU Max':<12} {'GPU Max':<12} {'Status'}")
print("-" * 60)

for name, op_func in ops:
    # CPU: works correctly
    with tf.device('/CPU:0'):
        # Conversion: torch.zeros -> tf.zeros. .T -> tf.transpose. .clone() -> implicit in TF op
        # Note: We use tf.Variable to mimic the mutable behavior of PyTorch tensors
        t_cpu = tf.Variable(tf.transpose(tf.zeros((50, 50))))
        op_func(t_cpu)
        cpu_max = tf.reduce_max(t_cpu).numpy()

    # GPU (MPS equivalent): fails silently in the original PyTorch bug
    # Check for GPU availability to avoid runtime errors if no GPU is present
    gpu_device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    with tf.device(gpu_device):
        t_gpu = tf.Variable(tf.transpose(tf.zeros((50, 50))))
        op_func(t_gpu)
        gpu_max = tf.reduce_max(t_gpu).numpy()

    status = "✓ OK" if gpu_max != 0.0 else "✗ BUG"
    print(f"{name:<20} {cpu_max:<12.4f} {gpu_max:<12.4f} {status}")
```