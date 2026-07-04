import torch
import numpy as np
import sys

# Handle environment issues preventing TensorFlow import (e.g., GLIBCXX version mismatch)
try:
    import tensorflow as tf
except ImportError as e:
    print(f"Skipping test: TensorFlow import failed due to environment issues (e.g., GLIBCXX version). Error: {e}")
    sys.exit(0)

def test_scatter_sum_with_no_grad():
    """
    Reproduces the logic of the original bug report's code snippet:
    
    with torch.no_grad():
        expert_counts.scatter_(1, topk_ids, 1)
        tokens_per_expert = expert_counts.sum(dim=0)
        
    Translated to TensorFlow, leveraging tf.keras.ops.add.
    """
    # Setup dimensions matching the issue description
    batch_size = 256
    num_experts = 64
    topk = 6
    
    # Initialize expert_counts (equivalent to new_zeros in the issue)
    expert_counts = tf.zeros((batch_size, num_experts), dtype=tf.int64)
    
    # Generate random topk_ids
    topk_ids = tf.random.uniform((batch_size, topk), minval=0, maxval=num_experts, dtype=tf.int64)
    
    # Prepare indices for scatter_nd_add
    # PyTorch scatter_(dim=1, index=topk_ids, value=1)
    # TF requires (batch, k, 2) indices for 2D tensor
    batch_indices = tf.range(batch_size)
    batch_indices = tf.tile(tf.expand_dims(batch_indices, 1), [1, topk])
    indices = tf.stack([batch_indices, topk_ids], axis=-1)
    
    # Create updates (adding 1)
    # Leveraging the similar API: tf.keras.ops.add
    zeros = tf.zeros((batch_size, topk), dtype=tf.int64)
    ones = tf.constant(1, dtype=tf.int64)
    updates = tf.keras.ops.add(zeros, ones)
    
    # Mimic torch.no_grad() context
    # In TensorFlow, we use tf.stop_gradient to prevent gradient flow
    with tf.GradientTape() as tape:
        # Perform scatter operation (equivalent to torch.ops.aten.scatter_.value)
        scattered_counts = tf.tensor_scatter_nd_add(expert_counts, indices, updates)
        
        # Apply stop_gradient to mimic the no_grad context
        scattered_counts = tf.stop_gradient(scattered_counts)
        
        # Perform sum operation (equivalent to torch.ops.aten.sum.dim_IntList)
        tokens_per_expert = tf.reduce_sum(scattered_counts, axis=0)
        
    # Verify the logic: The sum of all expert counts should equal the total number of updates
    total_updates = batch_size * topk
    total_sum = tf.reduce_sum(tokens_per_expert)
    
    assert total_sum == total_updates, f"Expected sum {total_updates}, got {total_sum}"
    assert tokens_per_expert.shape == (num_experts,), f"Expected shape ({num_experts},), got {tokens_per_expert.shape}"
    
    print("Test passed: Scatter and sum logic preserved with no_grad context.")

if __name__ == "__main__":
    test_scatter_sum_with_no_grad()