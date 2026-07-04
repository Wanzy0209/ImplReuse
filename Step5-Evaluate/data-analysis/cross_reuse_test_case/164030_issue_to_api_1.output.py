import tensorflow as tf
import numpy as np

def test_scatter_and_sum_with_tf_add():
    """
    Test case adapted from PyTorch issue 164030.
    Original logic: scatter values into a tensor, then sum along a dimension.
    This test uses tf.compat.v1.math.add to perform the summation logic.
    """
    # Setup dimensions matching the original issue
    batch_size = 256
    num_experts = 64
    topk = 6

    # Initialize expert_counts (zeros)
    expert_counts = tf.zeros([batch_size, num_experts], dtype=tf.int64)

    # Generate random topk_ids
    topk_ids = tf.random.uniform((batch_size, topk), minval=0, maxval=num_experts, dtype=tf.int32)

    # Create updates (all 1s)
    updates = tf.ones((batch_size, topk), dtype=tf.int64)

    # Perform scatter operation (equivalent to expert_counts.scatter_(1, topk_ids, 1))
    # tf.tensor_scatter_nd_add requires indices of shape (N, rank(tensor)).
    # Here rank(tensor) is 2. We need to construct (row, col) pairs.
    
    # 1. Flatten the topk_ids to get column indices
    flat_ids = tf.reshape(topk_ids, [-1])
    
    # 2. Create row indices corresponding to each batch element, repeated topk times
    batch_indices = tf.range(batch_size)
    batch_indices = tf.repeat(batch_indices, topk)
    
    # 3. Stack to get coordinates: shape (batch_size * topk, 2)
    indices = tf.stack([batch_indices, flat_ids], axis=1)
    
    # 4. Flatten updates to match the number of indices
    flat_updates = tf.reshape(updates, [-1])
    
    # 5. Perform scatter
    scattered_counts = tf.tensor_scatter_nd_add(expert_counts, indices, flat_updates)

    # Perform sum operation (equivalent to expert_counts.sum(dim=0))
    # To leverage the similar API (tf.compat.v1.math.add), we manually sum the tensor
    # by splitting it and adding the parts.
    
    # Split the batch into two halves
    split_size = batch_size // 2
    part1 = scattered_counts[:split_size, :]
    part2 = scattered_counts[split_size:, :]
    
    # Use tf.compat.v1.math.add to accumulate the counts
    # Naming it 'submod_1' to reference the naming collision issue in the bug report
    tokens_per_expert = tf.compat.v1.math.add(part1, part2, name="submod_1")

    # Verify the result against the standard reduce_sum
    expected_tokens = tf.reduce_sum(scattered_counts, axis=0)
    
    # Assertion
    assert tf.reduce_all(tf.equal(tokens_per_expert, expected_tokens)).numpy(), "Summation mismatch"

if __name__ == "__main__":
    test_scatter_and_sum_with_tf_add()
    print("Test passed.")