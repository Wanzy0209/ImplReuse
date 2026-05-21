import torch
import tensorflow as tf
import numpy as np
import pytest

def test_dynamic_cache_xla_compile():
    """
    Adapted from PyTorch Issue #167293.
    
    Original Issue: torch.export.export fails with torch._dynamo.exc.UserError: 
    Constraints violated (seq) when testing dynamic cache exportability with 
    varying sequence lengths.
    
    This test verifies the behavior of the similar TensorFlow API 
    `tf.xla.experimental.compile` when handling dynamic KV cache shapes 
    (specifically varying sequence lengths) across multiple runs.
    """
    
    # Define a computation that mimics a transformer step updating a KV cache.
    # This involves dynamic shape manipulation (concatenation) which is the 
    # source of the constraint violation in the PyTorch bug.
    def transformer_step(input_ids, past_key_values):
        # input_ids: [batch_size, 1]
        # past_key_values: A list of tensors, each [batch_size, seq_len, hidden_dim]
        
        new_caches = []
        # We project input_ids to hidden_dim to simulate the K/V projection
        # Using a simple tile/expand for minimal dependencies
        hidden_dim = tf.shape(past_key_values[0])[2]
        projected_input = tf.tile(input_ids[:, :, tf.newaxis], [1, 1, hidden_dim])
        projected_input = tf.cast(projected_input, tf.float32)

        for cache in past_key_values:
            # Update the cache by appending the new token.
            # This operation changes the sequence dimension dynamically.
            new_cache = tf.concat([cache, projected_input], axis=1)
            new_caches.append(new_cache)
        
        return new_caches

    batch_size = 1
    hidden_dim = 64
    num_layers = 2
    
    # The PyTorch error involved a constraint check on sequence length (seq <= 512).
    # We test multiple sequence lengths to verify the "exportability" (compilation)
    # across different dynamic shapes.
    sequence_lengths = [10, 128, 512, 600]

    for seq_len in sequence_lengths:
        # Create inputs for the current sequence length
        input_ids = tf.constant(np.random.randint(0, 1000, (batch_size, 1)), dtype=tf.int32)
        
        # Initialize past_key_values
        past_key_values = [
            tf.constant(np.random.randn(batch_size, seq_len, hidden_dim), dtype=tf.float32)
            for _ in range(num_layers)
        ]
        
        # Attempt to compile and run the computation using tf.xla.experimental.compile.
        # This is the semantic equivalent of torch.export.export in this context:
        # capturing the graph logic for the given inputs.
        try:
            # Note: tf.xla.experimental.compile compiles and runs immediately.
            # It validates shape consistency during the compilation phase.
            result = tf.xla.experimental.compile(
                transformer_step, 
                inputs=[input_ids, past_key_values]
            )
            
            # Assertions to verify correct execution
            assert len(result) == num_layers, "Should return cache for all layers"
            for i, cache in enumerate(result):
                # The sequence length should have increased by 1 (the new input token)
                expected_shape = (batch_size, seq_len + 1, hidden_dim)
                assert cache.shape == expected_shape, \
                    f"Layer {i} shape mismatch: expected {expected_shape}, got {cache.shape}"
                    
        except tf.errors.InvalidArgumentError as e:
            # In TensorFlow/XLA, shape mismatches or unsupported dynamic shape operations
            # often result in InvalidArgumentError. This corresponds to the 
            # "Constraints violated" UserError in PyTorch.
            pytest.fail(f"XLA Compilation failed for seq_len {seq_len} (Constraint Violation): {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error for seq_len {seq_len}: {e}")

if __name__ == "__main__":
    test_dynamic_cache_xla_compile()