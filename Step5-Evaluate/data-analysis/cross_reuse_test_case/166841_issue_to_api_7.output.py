import tensorflow as tf
import numpy as np
import sys

def test_relu_mixed_device_execution():
    """
    Test case for tf.keras.activations.relu inspired by PyTorch Issue 166841.
    
    The original bug involved a compiler (aoti_compile_and_package) incorrectly 
    assigning a CUDA kernel to a CPU operation in a mixed-device graph, leading 
    to a runtime device guard error.
    
    This test adapts that logic to TensorFlow. It verifies that tf.keras.activations.relu
    correctly handles mixed-device inputs (CPU and GPU) within a tf.function graph,
    ensuring that operations respect the device placement of their inputs and do not
    trigger device mismatch errors.
    """
    
    # Check for GPU availability to ensure the test is valid
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("Test skipped: No GPU available to test mixed-device execution.")
        return

    print("Starting mixed-device test for tf.keras.activations.relu...")

    # 1. Define a model structure similar to the bug report:
    #    - A component (buffer/variable) residing on CPU.
    #    - An input residing on GPU.
    class MixedDeviceModel(tf.Module):
        def __init__(self):
            super().__init__()
            # Register a buffer/variable on CPU
            with tf.device('/CPU:0'):
                self.cpu_data = tf.constant([-1.0, -2.0, 0.0, 3.0], dtype=tf.float32)

        # tf.function acts as the "compiler" here, tracing the graph
        @tf.function
        def __call__(self, gpu_input):
            # Operation on CPU tensors (mimicking the scatter_add_ in the bug)
            # We use the similar API: tf.keras.activations.relu
            cpu_result = tf.keras.activations.relu(self.cpu_data)

            # Operation on GPU tensors (mimicking the matmul in the bug)
            # We also use relu here to test the API's behavior on the GPU side
            gpu_result = tf.keras.activations.relu(gpu_input)

            return cpu_result, gpu_result

    # 2. Setup inputs
    # Create input on GPU
    with tf.device('/GPU:0'):
        gpu_input = tf.constant([-5.0, 0.0, 5.0, -1.0], dtype=tf.float32)

    model = MixedDeviceModel()

    # 3. Run the model
    # In the PyTorch bug, this step triggered:
    # "CUDAGuardImpl initialized with non-CUDA DeviceType: cpu"
    # We expect this to run without errors in TensorFlow.
    try:
        cpu_out, gpu_out = model(gpu_input)
    except tf.errors.InvalidArgumentError as e:
        print(f"Test FAILED: Device mismatch error occurred: {e}")
        sys.exit(1)

    # 4. Assertions
    # Verify correctness of the operation
    expected_cpu = [0.0, 0.0, 0.0, 3.0]
    expected_gpu = [0.0, 0.0, 5.0, 0.0]

    assert np.allclose(cpu_out.numpy(), expected_cpu), \
        f"CPU computation incorrect. Expected {expected_cpu}, got {cpu_out.numpy()}"
    
    assert np.allclose(gpu_out.numpy(), expected_gpu), \
        f"GPU computation incorrect. Expected {expected_gpu}, got {gpu_out.numpy()}"

    # Verify device placement
    # The bug was that the CPU op was treated as a CUDA op.
    # We explicitly check that the CPU result is actually on CPU.
    assert 'CPU' in cpu_out.device, \
        f"Device placement error: CPU result is on {cpu_out.device} instead of CPU."
    
    assert 'GPU' in gpu_out.device, \
        f"Device placement error: GPU result is on {gpu_out.device} instead of GPU."

    print("Test passed: tf.keras.activations_relu handled mixed devices correctly.")

if __name__ == "__main__":
    test_relu_mixed_device_execution()