import torch
import tensorflow as tf
import os

# Enable verbose logging to detect compilation/retracing events
# Similar to torch._logging.set_logs(recompiles=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
tf.get_logger().setLevel('INFO')

# Define a mock model computation to replace the FluxPipeline transformer
# This represents the workload we want to parallelize/compile
def model_computation(inputs):
    # Simulating a layer operation (e.g., a transformer block)
    w = tf.Variable(tf.random.normal([512, 512]), dtype=tf.float32)
    return tf.matmul(inputs, w)

# Setup inputs
# batch_parallel splits the 0-th dimension, so batch size must be divisible by num_shards
batch_size = 8
feature_dim = 512
inputs = [tf.random.normal([batch_size, feature_dim], dtype=tf.float32)]

# Apply the similar API: tf.compat.v1.tpu.batch_parallel
# This corresponds to the 'compile' step in the original bug report
num_shards = 2
try:
    # Note: batch_parallel is a graph-mode operation typically used with TPUs.
    # We construct the graph here.
    parallel_output = tf.compat.v1.tpu.batch_parallel(
        model_computation,
        inputs=inputs,
        num_shards=num_shards
    )

    # Execution
    # We use a tf.compat.v1.Session to run the graph, similar to pipe(...) in the original
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        print("Starting inference (Run 1)...")
        # Run 1: Check for initial compilation logs
        result_1 = sess.run(parallel_output)
        
        print("Starting inference (Run 2)...")
        # Run 2: Check for unexpected recompilation/retracing logs
        result_2 = sess.run(parallel_output)

        print(f"Execution successful. Output shape: {result_1.shape}")

except Exception as e:
    # This block handles cases where TPU hardware is not available,
    # ensuring the code structure is valid even if the environment lacks the specific hardware.
    print(f"Execution failed (expected if no TPU is available): {e}")