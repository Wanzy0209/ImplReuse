import torch
import tensorflow as tf
import numpy as np

# Enable eager execution for TensorFlow 2.x
tf.compat.v1.enable_eager_execution()

# Define the computation function (equivalent to SimpleMLP forward)
def computation(inputs):
    x = inputs[0]
    # Reshape input: (Batch, 28, 28) -> (Batch, 784)
    x = tf.reshape(x, [tf.shape(x)[0], 28 * 28])
    
    # Define weights using variable scope to allow reuse across calls
    with tf.compat.v1.variable_scope("simple_mlp", reuse=tf.compat.v1.AUTO_REUSE):
        # Layer 1: Linear(784, 128) + ReLU
        w1 = tf.compat.v1.get_variable("fc1_w", [784, 128], initializer=tf.random_normal_initializer())
        b1 = tf.compat.v1.get_variable("fc1_b", [128], initializer=tf.zeros_initializer())
        x = tf.nn.relu(tf.matmul(x, w1) + b1)
        
        # Layer 2: Linear(128, 10)
        w2 = tf.compat.v1.get_variable("fc2_w", [128, 10], initializer=tf.random_normal_initializer())
        b2 = tf.compat.v1.get_variable("fc2_b", [10], initializer=tf.zeros_initializer())
        x = tf.matmul(x, w2) + b2)
    return x

def main():
    # Attempt to initialize TPU (required for batch_parallel)
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)

        print("TPU initialized. Running test case...")

        # Use the strategy scope, analogous to torch.device("cuda")
        with strategy.scope():
            # Initialize variables by running the computation once
            _ = computation([tf.zeros([1, 28, 28])])

            # Setup Profiler (analogous to torch.profiler)
            tf.profiler.experimental.start("./log")

            # Execution loop
            for i in range(1, 11):
                # Create input tensor
                # Note: range(1, 11) avoids empty tensor issues for batch_parallel
                x = tf.random.normal([i, 28, 28])
                
                # Call the similar API: tf.compat.v1.tpu.batch_parallel
                # This shards the computation along the batch dimension
                y = tf.compat.v1.tpu.batch_parallel(
                    computation,
                    [x],
                    num_shards=8
                )
                
                # Calculate loss
                loss = tf.reduce_sum(y)
                
                # Force execution to ensure ops run
                _ = loss.numpy()

            tf.profiler.experimental.stop()
            print("Test case executed successfully.")

    except (ValueError, tf.errors.NotFoundError) as e:
        print(f"TPU not available or initialization failed: {e}")
        print("This test case is designed for a TPU environment to execute batch_parallel.")

if __name__ == "__main__":
    main()