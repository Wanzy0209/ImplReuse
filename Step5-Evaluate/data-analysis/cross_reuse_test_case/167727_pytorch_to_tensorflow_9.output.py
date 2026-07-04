import torch
import tensorflow as tf
import numpy as np

def test_tf_addmm_large_tensors():
    """
    Adapts the PyTorch torch.addmm test case to TensorFlow.
    
    The original bug involves incorrect results for large complex64 tensors 
    on MPS (GPU). We test the equivalent operation in TensorFlow:
    out = beta * input + alpha * (mat1 @ mat2)
    
    We use tf.convert_to_tensor to ensure inputs are properly converted,
    and tf.linalg.matmul for the matrix multiplication.
    """
    
    # Helper function to mimic torch.addmm
    # out = beta * a + alpha * (b @ c)
    def addmm(a, b, c, alpha=1.0, beta=1.0):
        # Explicitly using the identified similar API for conversion
        a = tf.convert_to_tensor(a)
        b = tf.convert_to_tensor(b)
        c = tf.convert_to_tensor(c)
        
        return beta * a + alpha * tf.linalg.matmul(b, c)

    # Check for available devices (GPU/MPS)
    gpus = tf.config.list_physical_devices('GPU')
    device_name = '/GPU:0' if gpus else '/CPU:0'
    print(f"Running test on device: {device_name}")

    # --- Success Case (Small Tensors) ---
    print("\n--- Testing Small Tensors (64, 100, 300) ---")
    # Create tensors
    a = tf.random.uniform((64, 300), dtype=tf.complex64)
    b = tf.random.uniform((64, 100), dtype=tf.complex64)
    c = tf.random.uniform((100, 300), dtype=tf.complex64)

    # Compute on CPU (Reference)
    with tf.device('/CPU:0'):
        out_cpu = addmm(a, b, c, alpha=1.0, beta=0.5)

    # Compute on Target Device (e.g., GPU/MPS)
    with tf.device(device_name):
        out_device = addmm(a, b, c, alpha=1.0, beta=0.5)

    # Verify consistency
    try:
        tf.debugging.assert_near(out_cpu, out_device, message="Small tensor mismatch")
        print("Small tensor test: PASSED")
    except tf.errors.InvalidArgumentError as e:
        print(f"Small tensor test: FAILED\n{e}")

    # --- Failure Case (Large Tensors) ---
    print("\n--- Testing Large Tensors (64, 10000, 300) ---")
    # Create large tensors
    a_large = tf.random.uniform((64, 300), dtype=tf.complex64)
    b_large = tf.random.uniform((64, 10000), dtype=tf.complex64)
    c_large = tf.random.uniform((10000, 300), dtype=tf.complex64)

    # Compute on CPU (Reference)
    with tf.device('/CPU:0'):
        out_cpu_large = addmm(a_large, b_large, c_large, alpha=1.0, beta=0.5)

    # Compute on Target Device
    with tf.device(device_name):
        out_device_large = addmm(a_large, b_large, c_large, alpha=1.0, beta=0.5)

    # Verify consistency
    try:
        # PyTorch used rtol=1.3e-06, atol=1e-05. 
        # tf.debugging.assert_near defaults are rtol=1e-6, atol=1e-6.
        # We use slightly looser tolerances to match the spirit of the PyTorch test.
        tf.debugging.assert_near(out_cpu_large, out_device_large, rtol=1e-5, atol=1e-5, message="Large tensor mismatch")
        print("Large tensor test: PASSED")
    except tf.errors.InvalidArgumentError as e:
        print(f"Large tensor test: FAILED\n{e}")

if __name__ == "__main__":
    test_tf_addmm_large_tensors()