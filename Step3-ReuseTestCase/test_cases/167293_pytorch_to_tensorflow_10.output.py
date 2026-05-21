import torch
import tensorflow as tf
import numpy as np
import pytest

def test_spatial_dropout_dynamic_timesteps():
    """
    Adapted from PyTorch issue #167293 (torch.export.export constraint violation).
    
    Original Bug: torch.export.export failed with 'Constraints violated (seq)' 
    when the sequence length changed between runs, violating inferred guards.
    
    This test verifies that tf.keras.layers.SpatialDropout1D handles dynamic 
    timesteps (sequence lengths) correctly when traced/compiled, ensuring 
    no similar constraint violations occur for the 'seq' dimension.
    """
    # 1. Define the model with SpatialDropout1D
    # Input shape: (Batch_Size, Timesteps, Channels). 
    # We set Timesteps to None to allow dynamic sequence lengths.
    inputs = tf.keras.Input(shape=(None, 32))
    # SpatialDropout1D drops entire 1D feature maps (channels) across the timestep dimension
    layer = tf.keras.layers.SpatialDropout1D(rate=0.5)
    outputs = layer(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # 2. "Export" / Trace the model
    # Using tf.function to trace the graph is analogous to torch.export.export.
    # It captures the computation logic based on input shapes.
    exported_model = tf.function(model.call)

    # 3. First Run: Establish initial shape context
    # PyTorch bug context: This run inferred a constraint (e.g., seq <= 512).
    seq_len_1 = 10
    input_data_1 = tf.random.normal((2, seq_len_1, 32))
    
    # Run in training mode to ensure dropout logic is active
    output_1 = exported_model(input_data_1, training=True)
    
    # Verify output shape matches input shape
    assert output_1.shape == (2, seq_len_1, 32), \
        f"Run 1 failed: Expected shape (2, {seq_len_1}, 32), got {output_1.shape}"

    # 4. Second Run: Change sequence length
    # PyTorch bug context: This run violated the constraint inferred in Run 1.
    # We test if the TensorFlow API handles this dynamic change gracefully.
    seq_len_2 = 20 # Different sequence length
    input_data_2 = tf.random.normal((2, seq_len_2, 32))

    try:
        # This should succeed without raising a constraint violation error
        output_2 = exported_model(input_data_2, training=True)
        
        # Verify output shape matches the new input shape
        assert output_2.shape == (2, seq_len_2, 32), \
            f"Run 2 failed: Expected shape (2, {seq_len_2}, 32), got {output_2.shape}"
            
        print("Test Passed: SpatialDropout1D handles dynamic timesteps correctly.")
        
    except Exception as e:
        pytest.fail(f"Test Failed: Constraint violation or error occurred with dynamic timesteps: {e}")

if __name__ == "__main__":
    test_spatial_dropout_dynamic_timesteps()