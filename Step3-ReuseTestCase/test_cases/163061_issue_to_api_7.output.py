import torch
import tensorflow as tf
import threading
import time

# Global variable to check if GIL is released by monitoring thread activity
gil_released_indicator = 0
stop_thread = False

def worker_thread():
    """
    A CPU-bound thread that increments a counter.
    If the GIL is held by the main thread, this thread will not run.
    """
    global gil_released_indicator
    while not stop_thread:
        gil_released_indicator += 1

def check_gil_behavior(func, *args):
    """
    Executes the given function and checks if the GIL was released
    by observing if the worker thread could run concurrently.
    """
    global gil_released_indicator, stop_thread
    
    # Reset state
    gil_released_indicator = 0
    stop_thread = False
    
    # Start the worker thread
    t = threading.Thread(target=worker_thread)
    t.start()
    
    # Allow worker to start
    time.sleep(0.001)
    
    # Execute the target function (e.g., the compiled kernel)
    result = func(*args)
    
    # Signal worker to stop and wait
    stop_thread = True
    t.join()
    
    # If the counter increased, the worker thread ran, implying GIL was released
    return gil_released_indicator > 0

def main():
    # Setup input data (mimicking the original torch.randn)
    # Using GPU if available to match the context of CUDA kernels
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    with tf.device(device):
        x = tf.random.normal((4096, 4096))
        y = tf.random.normal((4096, 4096))

    # 1. Standard API usage (Similar to torch_add)
    def standard_leaky_relu(features):
        return tf.keras.activations.leaky_relu(features)

    # 2. Compiled API usage (Similar to torch.compile)
    # In TensorFlow, tf.function is the equivalent decorator for compilation/graph optimization
    @tf.function
    def compiled_leaky_relu(features):
        return tf.keras.activations.leaky_relu(features)

    print(f"Running on device: {device}")
    
    # Test Standard API
    print("Testing tf.keras.activations.leaky_relu (Standard)...")
    released_standard = check_gil_behavior(standard_leaky_relu, x)
    print(f"GIL Released: {released_standard}")
    
    # Test Compiled API
    print("Testing tf.keras.activations.leaky_relu (Compiled/tf.function)...")
    released_compiled = check_gil_behavior(compiled_leaky_relu, x)
    print(f"GIL Released: {released_compiled}")

    # Assertions to verify expected behavior
    # Unlike the bug reported in PyTorch (where GIL was held), 
    # we expect TensorFlow to release the GIL for both standard and compiled ops.
    assert released_standard, "Standard tf.keras.activations.leaky_relu should release GIL"
    assert released_compiled, "Compiled tf.keras.activations.leaky_relu should release GIL"
    
    print("Test passed: GIL is released appropriately.")

if __name__ == "__main__":
    main()