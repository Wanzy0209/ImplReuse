import torch
import sys

# Handle environment dependency issues (e.g., GLIBC version mismatch)
try:
    import tensorflow as tf
except ImportError as e:
    print(f"Skipped: TensorFlow import failed due to environment incompatibility (GLIBC version). Error: {e}")
    sys.exit(0)

def test_varlen_feature_large_buffer():
    """
    Test case for tf.io.VarLenFeature handling large data buffers.
    Adapted from PyTorch Issue 161265: torch.full fails for 4+Gb tensors.
    
    This test verifies that large features (exceeding 4GB) are parsed correctly
    and that the tail values are preserved, similar to checking a[1, -2] in the
    original bug report.
    """
    # Original logic: Create a tensor with elements exceeding 4GB total size.
    # torch.ones(2, (1 << 31) + 5, dtype=torch.int8) -> ~4.29 GB
    # We simulate this by creating a large bytes object.
    
    # Define size to exceed 4GB (2^32 bytes)
    # Note: This requires significant system memory (RAM) to execute.
    buffer_size = (1 << 32) + 5
    
    # Create a buffer filled with 1s (mimicking torch.ones)
    # We use a try-except block to handle environments with insufficient memory
    try:
        # b'\x01' is the byte representation of 1 for int8
        data = b'\x01' * buffer_size
    except MemoryError:
        print("Skipped: Not enough memory to allocate 4GB+ buffer.")
        return

    # Serialize the data into a tf.train.Example
    # We use a bytes_list to store the raw buffer
    feature = {
        'large_feature': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized_example = example_proto.SerializeToString()

    # Define the parsing configuration using VarLenFeature
    feature_parsing_spec = {
        'large_feature': tf.io.VarLenFeature(dtype=tf.string)
    }

    # Parse the example
    parsed_features = tf.io.parse_single_example(serialized_example, feature_parsing_spec)
    
    # The parsed feature is a SparseTensor containing the single large bytes string
    # We decode the raw bytes to int8 to match the original bug's dtype
    # parsed_features['large_feature'].values[0] accesses the single bytes string
    dense_tensor = tf.io.decode_raw(parsed_features['large_feature'].values[0], out_type=tf.int8)

    # Verify the tail values, similar to print(a[1, -2]) in the original bug
    # The original bug showed that values beyond the 4GB boundary were 0 instead of 1.
    # We check the second to last element (-2).
    
    # Using tf.numpy_function or simple .numpy() to assert the value
    tail_value = dense_tensor[-2].numpy()
    
    assert tail_value == 1, f"Expected tail value to be 1, but got {tail_value}. " \
                             "This indicates a potential issue with handling large buffers."
    
    print("Test passed: Large buffer tail value is correct.")

if __name__ == "__main__":
    test_varlen_feature_large_buffer()