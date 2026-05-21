import torch
import tensorflow as tf

def test_tensor_operations_in_name_scope():
    """
    Adapts the PyTorch torch.compile bug reproduction logic to TensorFlow,
    utilizing tf.keras.name_scope as the context manager.
    
    The original bug involved torch.compile crashing on specific interleaving 
    of unsqueeze and cat operations. This test verifies that the equivalent 
    tensor manipulations execute correctly within the TensorFlow name scope.
    """
    
    # Define inputs matching the PyTorch test case
    # x: shape (1, 32), dtype int64
    # y: shape (1,), dtype int32
    x = tf.zeros((1, 32), dtype=tf.int64)
    y = tf.zeros((1,), dtype=tf.int32)

    # Cast y to int64 to mimic PyTorch's implicit type promotion during concatenation
    # (PyTorch promotes int32 and int64 to int64 during cat, TF requires matching dtypes)
    y = tf.cast(y, tf.int64)

    # Use tf.keras.name_scope as the similar API context
    with tf.keras.name_scope("bug_reproduction_164700"):
        # Logic: y2 = torch.cat([x[:, 1:], y[:, None] + 32 * 2048], dim=1)
        x_slice = x[:, 1:]
        y_expanded = tf.expand_dims(y, axis=1) + 32 * 2048
        y2 = tf.concat([x_slice, y_expanded], axis=1)

        # Logic: x2 = x[:, 1:, None]
        x2 = tf.expand_dims(x[:, 1:], axis=-1)

        # Logic: y3 = y2[:, -1:, None]
        y3 = tf.expand_dims(y2[:, -1:], axis=-1)

        # Logic: torch.cat([x2, y3], dim=1) + torch.arange(-2048, 0, device=device)[None, None, :]
        concat_xy = tf.concat([x2, y3], axis=1)
        
        # Create the range and broadcast it
        # PyTorch: torch.arange(-2048, 0, device=device)[None, None, :]
        arange = tf.range(-2048, 0, dtype=tf.int64)
        arange = tf.reshape(arange, (1, 1, -1))  # Shape (1, 1, 2048)
        
        result = concat_xy + arange

        # Logic: .reshape(1, 32 * 2048)
        output = tf.reshape(result, (1, 32 * 2048))

    # Verify the output shape matches expectations
    expected_shape = (1, 32 * 2048)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    print("Test passed: Operations executed successfully within tf.keras.name_scope.")

if __name__ == "__main__":
    test_tensor_operations_in_name_scope()