import torch
import tensorflow as tf

def test_issue_164030_replacement():
    """
    Test case generated based on Issue 164030 (HOP and pipelining naming collision).
    This test adapts the original reproduction logic using the similar API 
    (tf.io.FixedLenFeature) and replaces specific PyTorch calls with 
    tf.keras.activations.relu as requested.
    """

    # Use Similar API: tf.io.FixedLenFeature
    # Define the configuration for the input data, reflecting the structure
    # used in the original bug report (expert_counts and topk_ids).
    expert_feature = tf.io.FixedLenFeature([256, 64], tf.int64, default_value=0)
    ids_feature = tf.io.FixedLenFeature([256, 6], tf.int64, default_value=0)

    # Initialize tensors matching the FixedLenFeature specifications
    expert_counts = tf.zeros([256, 64], dtype=tf.int64)
    topk_ids = tf.constant([[1] * 6] * 256, dtype=tf.int64)

    # Original: with torch.no_grad():
    # Replacement: tf.keras.activations.relu
    # Since tf.keras.activations.relu is not a context manager, we apply it 
    # to the inputs to simulate the "context" modification or simply execute 
    # the replacement API call.
    # We cast to float32 as relu is typically used with floats, though it supports others.
    expert_counts = tf.keras.activations.relu(tf.cast(expert_counts, tf.float32))
    topk_ids = tf.keras.activations.relu(tf.cast(topk_ids, tf.float32))

    # Original: expert_counts.scatter_(1, topk_ids, 1)
    # Replacement: tf.keras.activations.relu
    # scatter_ is an in-place modification. relu returns a new tensor.
    # We apply the replacement API.
    expert_counts = tf.keras.activations.relu(expert_counts)

    # Original: tokens_per_expert = expert_counts.sum(dim=0)
    # Replacement: tf.keras.activations.relu
    # sum performs a reduction. relu performs an activation.
    # We apply the replacement API.
    tokens_per_expert = tf.keras.activations.relu(expert_counts)

    # Assertions to verify the test runs and produces output of expected shape
    # Note: The shapes are preserved because relu is element-wise, unlike sum.
    assert expert_counts.shape == (256, 64)
    assert tokens_per_expert.shape == (256, 64)
    
    print("Test passed.")

if __name__ == "__main__":
    test_issue_164030_replacement()