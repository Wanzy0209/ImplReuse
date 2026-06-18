import torch
import tensorflow as tf
import numpy as np

# Disable eager execution as tpu.rewrite is a graph-mode API
tf.compat.v1.disable_eager_execution()

# Define the computation function (similar to SimpleMLP forward)
def simple_mlp_forward(x):
    # Reshape input: (Batch, 28, 28) -> (Batch, 28*28)
    x = tf.reshape(x, [tf.shape(x)[0], 28 * 28])
    
    # FC1: Linear(28*28, 128) + ReLU
    # Using variables to simulate nn.Linear weights
    w1 = tf.compat.v1.get_variable("w1", shape=[28 * 28, 128], initializer=tf.random_normal_initializer())
    b1 = tf.compat.v1.get_variable("b1", shape=[128], initializer=tf.zeros_initializer())
    x = tf.nn.relu(tf.matmul(x, w1) + b1)
    
    # FC2: Linear(128, 10)
    w2 = tf.compat.v1.get_variable("w2", shape=[128, 10], initializer=tf.random_normal_initializer())
    b2 = tf.compat.v1.get_variable("b2", shape=[10], initializer=tf.zeros_initializer())
    x = tf.matmul(x, w2) + b2
    
    return x

def main():
    # Initialize TPU system
    # Note: This requires a TPU environment to run physically.
    # For the purpose of the test case structure, we include the standard initialization.
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)

    # Setup Profiler
    log_dir = "./log"
    tf.profiler.experimental.start(log_dir)

    # Define placeholder for input
    # Note: TPU rewrite prefers static shapes. We use a fixed batch size here 
    # to ensure the test case remains runnable and minimal, unlike the dynamic 
    # shapes in the PyTorch repro which might trigger XLA recompilation.
    x_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, 28, 28])

    # Use tf.device context to mimic torch.device("cuda")
    with tf.device("/device:TPU:0"):
        # Rewrite the computation for TPU
        # This is analogous to torch.compile with cpp_wrapper
        compiled_outputs = tf.compat.v1.tpu.rewrite(simple_mlp_forward, [x_ph])

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            
            # Run loop
            for i in range(10):
                # Create random input
                x_val = np.random.randn(1, 28, 28).astype(np.float32)
                
                # Execute compiled function
                y = sess.run(compiled_outputs, feed_dict={x_ph: x_val})
                
                # Calculate loss (sum)
                loss = np.sum(y)

    tf.profiler.experimental.stop()
    print("Test case executed. Check ./log for profiler traces.")

if __name__ == "__main__":
    main()