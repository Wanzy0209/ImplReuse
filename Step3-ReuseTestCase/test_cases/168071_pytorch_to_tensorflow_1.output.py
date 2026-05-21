import tensorflow as tf

# Adapt the original PyTorch test case to tf.nn.max_pool1d.
# Original PyTorch case: Input shape (6, 0), padding (0, 0, 0, 24).
# This corresponds to padding the last dimension (size 0) with (0, 0)
# and the second-to-last dimension (size 6) with (0, 24).

# tf.nn.max_pool1d expects a 3D input (Batch, Width, Channels) for 'NWC' format.
# We map the PyTorch dimensions (6, 0) to (Width, Channels) with a Batch size of 1.
# Input shape: (1, 6, 0)
input_tensor = tf.zeros((1, 6, 0))

# Explicit padding for 'NWC' format: [[0, 0], [pad_left, pad_right], [0, 0]]
# We pad the Width dimension (dim 1) by (0, 24) -> size becomes 30.
# We pad the Channels dimension (dim 2) by (0, 0) -> size remains 0.
# This mimics the logic of padding a 0-shape dimension with (0, 0).
explicit_padding = [[0, 0], [0, 24], [0, 0]]

try:
    # Use ksize=1 and strides=1 to isolate the padding behavior from pooling reduction.
    output = tf.nn.max_pool1d(
        input=input_tensor,
        ksize=1,
        strides=1,
        padding=explicit_padding,
        data_format='NWC'
    )
    
    print("Test passed.")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    
    # Expected shape: (1, 6 + 24, 0 + 0) -> (1, 30, 0)
    assert output.shape == (1, 30, 0), f"Expected shape (1, 30, 0), but got {output.shape}"

except Exception as e:
    print(f"Test failed with error: {e}")