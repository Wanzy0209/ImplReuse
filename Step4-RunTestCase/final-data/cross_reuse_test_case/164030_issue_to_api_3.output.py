import torch
import numpy as np

# Handle environment/dependency issues gracefully
TF_AVAILABLE = False
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import TensorFlow. Skipping test.")
    print(f"Error details: {e}")
    if "GLIBCXX" in str(e):
        print("Hint: This error is typically caused by an outdated system C++ library (libstdc++.so.6).")
        print("Please update your environment or install a compatible version of TensorFlow/glibc++.")

def test_add_layer_with_scatter_sum_logic():
    """
    Test case based on Issue 164030 logic (scatter and sum operations)
    adapted to use the similar API tf.keras.layers.add.
    
    The original issue involves a context manager (torch.no_grad) wrapping
    scatter and sum operations. This test replicates that structure using
    TensorFlow equivalents and utilizes tf.keras.layers.add.
    """
    
    if not TF_AVAILABLE:
        return

    # Setup inputs matching the dimensions in the bug report
    batch_size = 256
    num_experts = 64
    topk = 6
    
    # Initialize expert_counts (zeros) and topk_ids (random indices)
    expert_counts = tf.zeros((batch_size, num_experts), dtype=tf.int64)
    topk_ids = tf.random.uniform((batch_size, topk), minval=0, maxval=num_experts, dtype=tf.int64)
    
    # Use a context manager to mirror the 'with torch.no_grad():' structure
    # tf.name_scope is used here to group operations, relating to the naming 
    # aspect of the original bug (submod_i naming collisions).
    with tf.name_scope("hop_like_scope"):
        
        # 1. Replicate: expert_counts.scatter_(1, topk_ids, 1)
        # TensorFlow equivalent: tf.tensor_scatter_nd_add
        # We need to construct indices for the scatter operation
        batch_indices = tf.range(batch_size, dtype=tf.int64)
        batch_indices = tf.tile(tf.expand_dims(batch_indices, 1), [1, topk])
        batch_indices = tf.reshape(batch_indices, [-1])
        
        expert_indices = tf.reshape(topk_ids, [-1])
        
        # Stack to create (batch, expert) pairs
        scatter_indices = tf.stack([batch_indices, expert_indices], axis=1)
        updates = tf.ones([batch_size * topk], dtype=tf.int64)
        
        scattered_counts = tf.tensor_scatter_nd_add(expert_counts, scatter_indices, updates)
        
        # 2. Replicate: tokens_per_expert = expert_counts.sum(dim=0)
        # TensorFlow equivalent: tf.reduce_sum
        tokens_per_expert = tf.reduce_sum(scattered_counts, axis=0)
        
        # 3. Leverage Similar API: tf.keras.layers.add
        # The bug report involves wrapping operations. Here we use the Add layer
        # to combine the result with a bias or another tensor.
        add_layer = tf.keras.layers.Add(name="expert_accumulation")
        
        # Create a bias to add (e.g., initializing counts)
        bias = tf.zeros_like(tokens_per_expert)
        
        # Perform the addition
        result = add_layer([tokens_per_expert, bias])
        
        # Assertions to verify logic preservation
        # Total tokens should equal batch_size * topk
        expected_total = batch_size * topk
        actual_total = tf.reduce_sum(result).numpy()
        
        assert actual_total == expected_total, \
            f"Expected total tokens {expected_total}, but got {actual_total}"
        
        assert result.shape == (num_experts,), \
            f"Expected shape ({num_experts},), but got {result.shape}"

if __name__ == "__main__":
    test_add_layer_with_scatter_sum_logic()
    if TF_AVAILABLE:
        print("Test passed successfully.")