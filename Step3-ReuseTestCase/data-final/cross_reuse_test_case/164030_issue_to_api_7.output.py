import torch
import tensorflow as tf
import numpy as np

def test_tensorflow_scatter_sum_relu_context():
    """
    Test case based on Issue 164030 logic, translated to TensorFlow.
    
    Original Logic:
        with torch.no_grad():
            expert_counts.scatter_(1, topk_ids, 1)
            tokens_per_expert = expert_counts.sum(dim=0)
            
    This test preserves the scatter and sum operations within a context,
    and leverages the similar API (tf.nn.relu) to process the result.
    """
    # Setup inputs matching the dimensions in the original bug report
    # expert_counts: i64[256, 64]
    # topk_ids: i64[256, 6]
    batch_size = 256
    num_experts = 64
    topk_k = 6

    expert_counts = tf.zeros((batch_size, num_experts), dtype=tf.int64)
    # Generate random topk_ids within valid range
    topk_ids = tf.random.uniform((batch_size, topk_k), minval=0, maxval=num_experts, dtype=tf.int64)

    # Mimic the context manager usage (torch.no_grad)
    # In TensorFlow, we use tf.GradientTape to manage gradient context.
    # We watch the tensors to ensure the context is active, similar to how
    # the original code interacts with the autograd context.
    with tf.GradientTape() as tape:
        # Watch inputs to track operations (mimicking active graph context)
        tape.watch(expert_counts)
        tape.watch(topk_ids)

        # Mimic: expert_counts.scatter_(1, topk_ids, 1)
        # TF uses functional updates. We construct indices for tensor_scatter_nd_add.
        # We need to map (batch_idx, topk_idx) to flat indices or coordinate pairs.
        batch_indices = tf.range(batch_size)
        batch_indices = tf.repeat(batch_indices, topk_k)
        flat_topk_ids = tf.reshape(topk_ids, [-1])
        
        # Stack to create (N, 2) indices for the 2D tensor
        scatter_indices = tf.stack([batch_indices, flat_topk_ids], axis=1)
        updates = tf.ones((batch_size * topk_k,), dtype=tf.int64)
        
        # Perform scatter update
        scattered_counts = tf.tensor_scatter_nd_add(expert_counts, scatter_indices, updates)

        # Mimic: tokens_per_expert = expert_counts.sum(dim=0)
        tokens_per_expert = tf.reduce_sum(scattered_counts, axis=0)

        # Leverage Similar API: tf.nn.relu
        # The original bug report highlights context handling. 
        # We apply tf.nn.relu (which uses context.get_default() internally) 
        # to the result to verify the operation integrates correctly within the context.
        # Cast to float32 as relu typically operates on floats.
        activated_tokens = tf.nn.relu(tf.cast(tokens_per_expert, tf.float32))

    # Assertions
    # Check output shape
    assert activated_tokens.shape == (num_experts,), f"Expected shape ({num_experts},), got {activated_tokens.shape}"
    
    # Since we only added positive counts, ReLU should be the identity function here.
    # We verify that the values are non-negative and match the sum.
    expected_sum = tf.cast(tf.reduce_sum(scattered_counts, axis=0), tf.float32)
    assert tf.reduce_all(tf.equal(activated_tokens, expected_sum)), "ReLU output should match input for non-negative values"
    
    print("Test passed: Scatter, Sum, and ReLU operations executed successfully within context.")

if __name__ == "__main__":
    test_tensorflow_scatter_sum_relu_context()