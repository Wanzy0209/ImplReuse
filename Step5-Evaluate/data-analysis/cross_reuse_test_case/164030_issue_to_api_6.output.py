import tensorflow as tf
import numpy as np

# Fix: Enable eager execution to ensure .numpy() method is available on Tensors.
# This is required because the test logic relies on eager execution behavior,
# but the environment might be defaulting to Graph Mode (TF 1.x behavior).
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

def test_sparse_accumulation_and_sum():
    """
    Test case adapted from PyTorch issue 164030 logic.
    
    Original Logic:
    The issue demonstrates a code block where values are scattered into a buffer
    (accumulated) and then summed along a dimension.
    
    Adaptation:
    This test leverages the similar API `tf.sparse.add` to perform the accumulation
    logic. Instead of scattering into a dense buffer, we construct sparse tensors
    representing the state and the updates, add them together, and then perform
    a reduction sum, mirroring the original intent.
    """
    
    # 1. Setup inputs mimicking the original 'expert_counts' and 'topk_ids'
    # We use sparse tensors to represent the data efficiently.
    
    # Base sparse tensor (initial state of expert_counts)
    # Shape: [3, 4] (Batch size 3, 4 experts)
    base_indices = np.array([[0, 1], [1, 2]], dtype=np.int64)
    base_values = np.array([1.0, 1.0], dtype=np.float32)
    base_shape = [3, 4]
    sp_base = tf.sparse.SparseTensor(base_indices, base_values, base_shape)

    # Updates sparse tensor (mimicking the scatter operation with value 1)
    # In the original: expert_counts.scatter_(1, topk_ids, 1)
    # We define indices where we want to add 1.0
    update_indices = np.array([[0, 1], [2, 0]], dtype=np.int64) 
    # Note: [0, 1] overlaps with base to test accumulation
    update_values = np.array([1.0, 1.0], dtype=np.float32)
    sp_updates = tf.sparse.SparseTensor(update_indices, update_values, base_shape)

    # 2. Perform accumulation using the similar API: tf.sparse.add
    # This replaces the in-place scatter_ operation logic.
    # It adds the values from sp_updates to sp_base.
    accumulated = tf.sparse.add(sp_base, sp_updates)

    # 3. Perform summation using tf.sparse.reduce_sum
    # This replaces expert_counts.sum(dim=0) from the original code.
    # Summing over dimension 0 (the batch dimension) to get tokens per expert.
    result = tf.sparse.reduce_sum(accumulated, axis=0)

    # 4. Assertions
    # Expected accumulation logic:
    # [0,1] -> 1.0 (base) + 1.0 (update) = 2.0
    # [1,2] -> 1.0 (base) + 0.0 (update) = 1.0
    # [2,0] -> 0.0 (base) + 1.0 (update) = 1.0
    
    # Sum over dim 0 (columns):
    # Col 0: 1.0 (from [2,0])
    # Col 1: 2.0 (from [0,1])
    # Col 2: 1.0 (from [1,2])
    # Col 3: 0.0
    
    # tf.sparse.reduce_sum returns a dense Tensor by default
    expected_result = tf.constant([1.0, 2.0, 1.0, 0.0], dtype=tf.float32)
    
    # Verify the result matches the expected accumulation and summation
    assert tf.reduce_all(tf.equal(result, expected_result)).numpy(), \
        f"Expected {expected_result.numpy()}, but got {result.numpy()}"

if __name__ == "__main__":
    test_sparse_accumulation_and_sum()
    print("Test passed successfully.")