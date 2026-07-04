import os
import time
import sys

# Attempt to import TensorFlow
# The error indicates a system library incompatibility (GLIBCXX).
# We handle this gracefully to prevent a crash.
try:
    import tensorflow as tf
except ImportError as e:
    print(f"Error importing TensorFlow: {e}")
    print("Skipping test due to missing or incompatible dependencies (GLIBCXX).")
    print("Please ensure your environment has the required C++ libraries for TensorFlow.")
    sys.exit(1)

# List of threads to test
threads_list = [1, 2, 4, 8, 16, 32, 48]

# Size of the large tensors
# Using large dimensions to ensure the operation is computationally heavy
batch_size = 10000
num_classes = 10000

# Store results
times = []

# Benchmark for each thread count
for threads in threads_list:
    # Set environment variables and thread count
    # TensorFlow respects these environment variables for underlying BLAS/MKL libraries
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(threads)

    # Configure TensorFlow threading
    # This is the TensorFlow equivalent to torch.set_num_threads
    tf.config.threading.set_intra_op_parallelism_threads(int(threads))
    tf.config.threading.set_inter_op_parallelism_threads(int(threads))

    # Create random tensors
    # Logits: unnormalized log probabilities, shape [batch_size, num_classes]
    logits = tf.random.normal((batch_size, num_classes))
    # Labels: class indices, shape [batch_size]
    labels = tf.random.uniform((batch_size,), maxval=num_classes, dtype=tf.int32)

    # Warm up
    _ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

    # Time the operation
    start_time = time.time()
    _ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    elapsed_time = time.time() - start_time

    print(f"Threads: {threads}, Time: {elapsed_time:.4f} s")
    times.append(elapsed_time)