import torch
import tensorflow as tf
import pytest

def test_tpu_rewrite_constraint_violation():
    """
    Adapts the PyTorch torch.export.export constraint violation test to 
    tf.compat.v1.tpu.rewrite.
    
    The original bug involves a sequence length (seq) violating a constraint (seq <= 512)
    during the export/compilation process. This test verifies that the TensorFlow TPU 
    rewrite API correctly identifies or fails when input constraints are violated 
    during the graph execution.
    """
    
    # Define the computation function (mimicking the transformer cache logic)
    def cache_update_computation(past_key_values, new_tokens):
        # past_key_values: [batch, cache_len, dim]
        # new_tokens: [batch, seq_len, dim]
        
        # Concatenate to update cache
        updated_cache = tf.concat([past_key_values, new_tokens], axis=1)
        
        # Get the new sequence length
        current_seq_len = tf.shape(updated_cache)[1]
        
        # Enforce the constraint: seq <= 512
        # This mimics the guard/check that failed in PyTorch:
        # "Not all values of seq ... in the specified range seq <= 512"
        tf.debugging.assert_less_equal(
            current_seq_len, 
            512, 
            message="Constraints violated (seq)! Not all values of seq in the specified range seq <= 512."
        )
        
        return updated_cache

    # Setup inputs
    # We simulate a scenario where the sequence length exceeds the constraint.
    # Initial cache length: 500
    # New tokens length: 20
    # Total: 520 (> 512)
    
    batch_size = 1
    hidden_dim = 64
    cache_len = 500
    seq_len = 20
    
    # Create tensors
    past_key_values = tf.zeros((batch_size, cache_len, hidden_dim), dtype=tf.float32)
    new_tokens = tf.ones((batch_size, seq_len, hidden_dim), dtype=tf.float32)
    
    # The PyTorch error occurs during export/compilation or immediately upon guard check.
    # In TF, tpu.rewrite compiles the graph and executes it. 
    # We expect the execution of the rewritten function to fail due to the assertion.
    
    try:
        # Initialize TPU system (required for tpu.rewrite)
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        tpu_strategy = tf.distribute.TPUStrategy(resolver)
        
        with tpu_strategy.scope():
            # Call the API: tf.compat.v1.tpu.rewrite
            # This compiles the computation for TPU and executes it with the provided inputs.
            # We expect an InvalidArgumentError due to the constraint violation.
            with pytest.raises(tf.errors.InvalidArgumentError) as exc_info:
                tf.compat.v1.tpu.rewrite(
                    computation=cache_update_computation,
                    inputs=[past_key_values, new_tokens]
                )
            
            # Verify the error message matches the constraint violation logic
            assert "Constraints violated" in str(exc_info.value) or "512" in str(exc_info.value)
            
    except (ValueError, tf.errors.NotFoundError):
        # Fallback for non-TPU environments to ensure the test code is syntactically valid
        # and demonstrates the logic, even if it can't run the XLA compilation.
        pytest.skip("TPU hardware not available for tf.compat.v1.tpu.rewrite test.")

if __name__ == "__main__":
    test_tpu_rewrite_constraint_violation()