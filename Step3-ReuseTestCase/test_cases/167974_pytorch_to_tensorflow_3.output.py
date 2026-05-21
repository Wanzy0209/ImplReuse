import torch
import tensorflow as tf
import numpy as np

def test_dropout_with_2d_input():
    """
    Adapted test case for tf.keras.backend.dropout based on the 
    torch.nn.EmbeddingBag bug report (Issue ID: 167974).
    
    The original bug involved incorrect handling of offsets when 
    include_last_offset was True and input was 2D.
    
    Here we verify that tf.keras.backend.dropout handles the 2D input 
    structure correctly and respects the provided parameters (rate, seed).
    """
    
    # Input tensor matching the structure from the PyTorch bug report
    # Original: torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=torch.long)
    # Note: Dropout typically operates on floats, so we cast to float32.
    input_tensor = tf.constant([[1.0, 2.0, 4.0, 5.0], [4.0, 3.0, 2.0, 9.0]], dtype=tf.float32)
    
    # Parameters
    # In the original bug, 'include_last_offset=True' was the critical flag.
    # For dropout, we verify behavior with a specific rate and seed.
    rate = 0.5
    seed = 42
    
    # Execute the API
    # The provided snippet for tf.keras.backend.dropout indicates that seed 
    # might be mandatory in certain contexts (e.g., DTensor), so we provide it.
    output = tf.keras.backend.dropout(input_tensor, rate, seed=seed)
    
    # Verification
    # 1. Check shape preservation: The output should maintain the 2D structure of the input.
    assert output.shape == input_tensor.shape, \
        f"Shape mismatch: expected {input_tensor.shape}, got {output.shape}"
        
    # 2. Check operation effect: 
    # Dropout scales the remaining inputs by 1/(1-rate). With rate=0.5, scale is 2.0.
    # Therefore, the output should generally be scaled or zeroed, not identical to input.
    # We check if the output is different from the input (ignoring the exact stochastic mask).
    # Since scale is 2.0, output will never equal input unless rate is 0.
    assert not tf.reduce_all(tf.equal(input_tensor, output)).numpy(), \
        "Output should differ from input due to dropout scaling/masking."
        
    print("Test passed: tf.keras.backend.dropout handled 2D input correctly.")

if __name__ == "__main__":
    test_dropout_with_2d_input()