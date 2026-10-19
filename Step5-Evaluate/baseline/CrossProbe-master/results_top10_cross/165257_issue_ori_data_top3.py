```python
import tensorflow as tf

# Check for GPU availability to mimic the CPU vs MPS comparison
# Note: TensorFlow uses 'GPU' generally, on Mac it utilizes Metal via MPS backend automatically if configured.
gpus = tf.config.list_physical_devices('GPU')
has_gpu = len(gpus) > 0

# Define operations
# Conversion: PyTorch in-place random ops (_) are replaced by TF functional ops returning new tensors.
ops = [
    ("normal_(0,1)", lambda t: tf.random.normal(tf.shape(t), mean=0, stddev=1)),
    ("uniform_(0,1)", lambda t: tf.random.uniform(tf.shape(t), minval=0, maxval=1)),
    ("exponential_(1)", lambda t: tf.random.gamma(tf.shape(t), alpha=1.0)), # Gamma(1) is Exp(1)
    ("bernoulli_(0.5)", lambda t: tf.cast(tf.random.uniform(tf.shape(t)) < 0.5, t.dtype)),
    ("random_()", lambda t: tf.random.uniform(tf.shape(t))),
    ("random_(10)", lambda t: tf.random.uniform(tf.shape(t), maxval=10, dtype=tf.int32)),
    ("random_(0,10)", lambda t: tf.random.uniform(tf.shape(t), minval=0, maxval=10, dtype=tf.int32)),
]

print(f"{'Operation':<20} {'CPU Max':<12} {'GPU Max':<12} {'Status'}")
print("-" * 60)

for name, op_func in ops:
    # CPU: works correctly
    with tf.device('/CPU:0'):
        t_cpu = tf.zeros((50, 50))
        # Conversion: .T.clone() creates a non-contiguous copy. tf.transpose is the TF equivalent.
        t_cpu = tf.transpose(t_cpu)
        t_cpu = op_func(t_cpu)
        cpu_max = float(tf.reduce_max(t_cpu).numpy())

    # GPU: Check if available (MPS equivalent on Mac)
    if has_gpu:
        with tf.device('/GPU:0'):
            t_gpu = tf.zeros((50, 50))
            t_gpu = tf.transpose(t_gpu)
            t_gpu = op_func(t_gpu)
            gpu_max = float(tf.reduce_max(t_gpu).numpy())
    else:
        gpu_max = 0.0
        # If no GPU, we can't test the MPS bug, but we keep the structure
        # print("Note: No GPU detected, skipping GPU test.")

    # Status: Check if GPU operation produced non-zero results (mimicking the bug check)
    status = "✓ OK" if gpu_max != 0.0 else "✗ BUG"
    print(f"{name:<20} {cpu_max:<12.4f} {gpu_max:<12.4f} {status}")
```