import tensorflow as tf
import numpy as np

def test_sparse_segment_mean_indices_dtype():
    """
    Adapted test case based on PyTorch Issue 166042.
    The original bug involved an assertion failure: assert "int" in str(indices.get_dtype()).
    This occurred because the fuzzer generated non-integer (bfloat16) indices for an embedding operation.
    
    This test verifies the behavior of tf.compat.v1.sparse_segment_mean when provided
    with bfloat16 indices (mimicking the fuzzer input) versus valid integer indices.
    """
    
    # Setup data tensor (equivalent to 'weight' in PyTorch embedding)
    # Shape: (5, 3)
    data = tf.constant(np.random.rand(5, 3), dtype=tf.float32)

    # --- Case 1: Valid Integer Indices ---
    # PyTorch embedding expects indices to be int64 or int32.
    # TensorFlow sparse_segment_mean expects int32 or int64.
    print("Testing with valid int32 indices...")
    indices_int32 = tf.constant([0, 1, 2, 3, 4], dtype=tf.int32)
    segment_ids_int32 = tf.constant([0, 0, 1, 1, 2], dtype=tf.int32)
    
    try:
        result_int32 = tf.compat.v1.sparse_segment_mean(data, indices_int32, segment_ids_int32)
        print("Success: Operation completed with int32 indices.")
        # print(result_int32.numpy())
    except Exception as e:
        print(f"Failure: {e}")

    # --- Case 2: Bfloat16 Indices (Fuzzer/Bug Scenario) ---
    # The original PyTorch bug was triggered because the code checked if "int" was in the dtype string.
    # Since "bfloat16" does not contain "int", the assertion failed.
    # We test if TensorFlow handles this invalid type gracefully (e.g., raising a clear error).
    print("\nTesting with bfloat16 indices (simulating fuzzer input)...")
    indices_bf16 = tf.constant([0, 1, 2, 3, 4], dtype=tf.bfloat16)
    segment_ids_bf16 = tf.constant([0, 0, 1, 1, 2], dtype=tf.bfloat16)
    
    try:
        # TensorFlow ops generally require int32/int64 for indices.
        # Passing bfloat16 should result in a TypeError.
        result_bf16 = tf.compat.v1.sparse_segment_mean(data, indices_bf16, segment_ids_bf16)
        print("Unexpected: Operation completed with bfloat16 indices.")
    except TypeError as e:
        print(f"Expected Behavior: TypeError raised for bfloat16 indices.")
        # print(f"Details: {e}")
    except Exception as e:
        print(f"Exception raised: {e}")

if __name__ == "__main__":
    test_sparse_segment_mean_indices_dtype()