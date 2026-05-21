import torch
import tensorflow as tf
import gc

# Ensure eager execution is enabled at the start of the program
tf.compat.v1.enable_eager_execution()

def count_tensors():
    """
    Helper function to count the number of live TensorFlow tensors.
    This mimics the behavior observed in the PyTorch bug report where
    the number of live tensors is monitored to detect memory leaks.
    """
    gc.collect()
    return sum(1 for obj in gc.get_objects() if isinstance(obj, tf.Tensor))

def test_memory_leak_in_eager_execution():
    """
    Adapted test case to check for memory leaks when running a loop
    in TensorFlow eager execution mode, mirroring the logic of the
    original torch.compile memory leak report.
    """
    # Define a simple model to simulate the workload.
    # Note: The original bug involved flash_attn_varlen_func. 
    # We use a standard Dense layer here to ensure the test is runnable 
    # without external dependencies, while preserving the loop structure.
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Simulation parameters
    batch_size = 32
    input_dim = 10
    steps = 300
    log_interval = 50

    print(f"Starting test for {steps} steps...")
    print("-" * 80)

    for step in range(steps):
        # Generate random input data
        x = tf.random.normal((batch_size, input_dim))
        y = tf.random.normal((batch_size, 10))

        # Forward pass
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = tf.reduce_mean(tf.square(logits - y))

        # Backward pass
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Monitor memory usage at intervals
        if (step + 1) % log_interval == 0:
            num_tensors = count_tensors()
            # Mimicking the output format of the original bug report
            print(f"Step {(step + 1):4d} | Loss: {loss.numpy():.4f} | Tensors: {num_tensors}")
            
            # In the original bug, the assertion would be that Tensors count 
            # grows indefinitely. Here we just log it for verification.
            # If this were a regression test for a leak, we might assert:
            # assert num_tensors < threshold, "Memory leak detected!"

    print("-" * 80)
    print("Test completed.")

if __name__ == "__main__":
    test_memory_leak_in_eager_execution()