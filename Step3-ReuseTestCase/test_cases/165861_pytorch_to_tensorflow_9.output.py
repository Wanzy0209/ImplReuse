import torch
import tensorflow as tf
import numpy as np

def test_tf_one_hot_large_dimensions():
    """
    Adapted test case to check if tf.one_hot handles dimensions larger than 
    uint16 max (2**16) correctly, similar to the PyTorch reflect padding bug.
    """
    
    # Check for GPU availability to match the CUDA context of the original bug
    gpus = tf.config.list_physical_devices('GPU')
    device = '/GPU:0' if gpus else '/CPU:0'
    
    print(f"Running test on device: {device}")

    with tf.device(device):
        # Case 1: Batch dimension is exactly 2**16 (65536)
        # Mimics: x = torch.rand(2**16, 2, device="cuda")
        print("\n--- Test Case 1: Batch dimension = 2**16 ---")
        try:
            indices = tf.zeros((2**16, 2), dtype=tf.int32)
            depth = 10
            result = tf.one_hot(indices, depth)
            print(f"Success. Output shape: {result.shape}")
        except Exception as e:
            print(f"FAILED. Error: {e}")

        # Case 2: Batch dimension is 2**16 + 1 (Just over the limit)
        # Mimics: x = torch.rand(2**16 + 1, 2, device="cuda")
        print("\n--- Test Case 2: Batch dimension = 2**16 + 1 ---")
        try:
            indices = tf.zeros((2**16 + 1, 2), dtype=tf.int32)
            depth = 10
            result = tf.one_hot(indices, depth)
            print(f"Success. Output shape: {result.shape}")
        except Exception as e:
            print(f"FAILED. Error: {e}")

        # Case 3: Depth (new dimension) is 2**16
        # Checks if the generated dimension size causes overflow
        print("\n--- Test Case 3: Depth = 2**16 ---")
        try:
            indices = tf.constant([0, 1], dtype=tf.int32)
            depth = 2**16
            result = tf.one_hot(indices, depth)
            print(f"Success. Output shape: {result.shape}")
        except Exception as e:
            print(f"FAILED. Error: {e}")

        # Case 4: Depth is 2**16 + 1
        print("\n--- Test Case 4: Depth = 2**16 + 1 ---")
        try:
            indices = tf.constant([0, 1], dtype=tf.int32)
            depth = 2**16 + 1
            result = tf.one_hot(indices, depth)
            print(f"Success. Output shape: {result.shape}")
        except Exception as e:
            print(f"FAILED. Error: {e}")

if __name__ == "__main__":
    test_tf_one_hot_large_dimensions()