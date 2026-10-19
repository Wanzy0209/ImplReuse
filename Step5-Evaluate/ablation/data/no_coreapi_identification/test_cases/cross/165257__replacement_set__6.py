import torch
import tensorflow as tf

# Test each operation on CPU and GPU (simulating MPS)
# We use tf.control_dependencies to wrap operations and checks
# We use torch.dist to verify the results

# Define operations using TensorFlow, wrapped in tf.control_dependencies
# Replacing normal_, uniform_, etc with tf.control_dependencies
ops = [
    ("normal_(0,1)", lambda t: tf.control_dependencies([t.assign(tf.random.normal(t.shape))])),
    ("uniform_(0,1)", lambda t: tf.control_dependencies([t.assign(tf.random.uniform(t.shape))])),
    ("exponential_(1)", lambda t: tf.control_dependencies([t.assign(tf.random.gamma(t.shape, 1.0))])),
    ("bernoulli_(0.5)", lambda t: tf.control_dependencies([t.assign(tf.cast(tf.random.uniform(t.shape) < 0.5, t.dtype))])),
    ("random_()", lambda t: tf.control_dependencies([t.assign(tf.random.uniform(t.shape))])),
    ("random_(10)", lambda t: tf.control_dependencies([t.assign(tf.random.uniform(t.shape, maxval=10))])),
    ("random_(0,10)", lambda t: tf.control_dependencies([t.assign(tf.random.uniform(t.shape, minval=0, maxval=10))])),
]

print(f"{'Operation':<20} {'CPU Dist':<12} {'GPU Dist':<12} {'Status'}")
print("-" * 60)

for name, op_func in ops:
    # CPU: works correctly
    with tf.device("/CPU:0"):
        # Create a non-contiguous tensor (transpose)
        # Replacing T.clone with tf.control_dependencies
        # We wrap the transpose operation to ensure it's handled
        t_cpu = tf.Variable(tf.zeros((50, 50)))
        with tf.control_dependencies([t_cpu]):
            t_cpu_t = tf.transpose(t_cpu)
            
            # Apply operation
            # Replacing normal_ with tf.control_dependencies
            with op_func(t_cpu_t):
                # Check result
                # Replacing t_cpu.max with tf.control_dependencies
                # We wrap the read operation to ensure it happens after assignment
                with tf.control_dependencies([t_cpu_t.read_value()]):
                    # Replacing torch.zeros with torch.dist
                    # We use torch.dist to check distance from zero
                    val = t_cpu_t.numpy()
                    t_val = torch.from_numpy(val)
                    # Use a scalar zero tensor for distance calculation
                    t_zero = torch.tensor(0.0)
                    cpu_dist = torch.dist(t_val, t_zero).item()

    # GPU: test
    with tf.device("/GPU:0"):
        t_gpu = tf.Variable(tf.zeros((50, 50)))
        with tf.control_dependencies([t_gpu]):
            t_gpu_t = tf.transpose(t_gpu)
            
            with op_func(t_gpu_t):
                with tf.control_dependencies([t_gpu_t.read_value()]):
                    val = t_gpu_t.numpy()
                    t_val = torch.from_numpy(val)
                    t_zero = torch.tensor(0.0)
                    gpu_dist = torch.dist(t_val, t_zero).item()

    status = " OK" if gpu_dist != 0.0 else " BUG"
    print(f"{name:<20} {cpu_dist:<12.4f} {gpu_dist:<12.4f} {status}")