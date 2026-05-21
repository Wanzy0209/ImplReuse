import tensorflow as tf

def test_tf_get_memory_usage():
    """
    Test case for tf.config.experimental.get_memory_usage.
    
    Context: The original issue (Issue 163337) involved a compilation error on ROCm 
    when using PyTorch's cpp_extension.load. While this TensorFlow API does not 
    compile C++ extensions, it interacts with the GPU memory layer. This test 
    verifies the functionality of the similar API in a GPU environment.
    """
    # Check for GPU availability (ROCm or CUDA)
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("Test skipped: No GPU devices found.")
        return

    # Target the first available GPU
    device_name = gpus[0].name # e.g., '/physical_device:GPU:0'
    
    print(f"Querying memory usage for device: {device_name}")

    # Call the API
    try:
        # Note: tf.config.experimental.get_memory_usage is deprecated in favor of
        # get_memory_info, but we are testing the specific similar API requested.
        memory_usage = tf.config.experimental.get_memory_usage(device_name)
    except Exception as e:
        print(f"API call failed: {e}")
        raise

    # Verify behavior
    # The API returns memory in bytes as an integer or float
    assert isinstance(memory_usage, (int, float)), \
        f"Expected numeric return type, got {type(memory_usage)}"
    
    assert memory_usage >= 0, \
        f"Memory usage cannot be negative, got {memory_usage}"

    print(f"Success: Current memory usage is {memory_usage} bytes")

if __name__ == "__main__":
    test_tf_get_memory_usage()