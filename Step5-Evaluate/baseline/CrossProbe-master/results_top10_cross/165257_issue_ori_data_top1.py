```python
import tensorflow as tf

# Conversion: PyTorch in-place random ops map to tf.Variable.assign with TF functional random ops
ops = [
    ("normal_(0,1)", lambda v: v.assign(tf.random.normal(v.shape, mean=0.0, stddev=1.0))),
    ("uniform_(0,1)", lambda v: v.assign(tf.random.uniform(v.shape, minval=0.0, maxval=1.0))),
    ("exponential_(1)", lambda v: v.assign(tf.random.exponential(v.shape, lam=1.0))),
    ("bernoulli_(0.5)", lambda v: v.assign(tf.cast(tf.random.uniform(v.shape) < 0.5, tf.float32))),
    ("random_()", lambda v: v.assign(tf.random.uniform(v.shape))),
    ("random_(10)", lambda v: v.assign(tf.random.uniform(v.shape, minval=0.0, maxval=10.0))),
    ("random_(0,10)", lambda v: v.assign(tf.random.uniform(v.shape, minval=0.0, maxval=10.0))),
]

print(f"{'Operation':<20} {'CPU Max':<12} {'GPU Max':<12} {'Status'}")
print("-" * 60)

# Conversion: Check for GPU availability to replace MPS
gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
device_name = '/GPU:0' if gpu_available else '/CPU:0'

for name, op_func in ops:
    # CPU: works correctly
    with tf.device('/CPU:0'):
        # Conversion: Mimic non-contiguous tensor creation (transpose)
        t_cpu = tf.Variable(tf.transpose(tf.zeros((50, 50), dtype=tf.float32)))
        op_func(t_cpu)
        cpu_max = tf.reduce_max(t_cpu).numpy()

    # GPU: TensorFlow equivalent of MPS
    with tf.device(device_name):
        # Conversion: Mimic non-contiguous tensor creation (transpose)
        t_gpu = tf.Variable(tf.transpose(tf.zeros((50, 50), dtype=tf.float32)))
        op_func(t_gpu)
        gpu_max = tf.reduce_max(t_gpu).numpy()

    status = "✓ OK" if gpu_max != 0.0 else "✗ BUG"
    print(f"{name:<20} {cpu_max:<12.4f} {gpu_max:<12.4f} {status}")
```