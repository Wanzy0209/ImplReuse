import tensorflow as tf

def test_negative_padding_behavior():
    """
    Adapts the PyTorch constant_pad_nd negative padding test case to 
    tf.nn.max_pool1d to verify behavior with negative explicit padding.
    """
    # Create a 3D input tensor for max_pool1d (Batch, Width, Channels)
    # Corresponds to the PyTorch input shape [5, 3] adapted to [5, 3, 1]
    input_tensor = tf.ones([5, 3, 1], dtype=tf.float32)

    # Define explicit padding with negative values.
    # PyTorch case: padding=[-1, -2] applied to the last dimension (size 3).
    # TensorFlow case (NWC format): [[0, 0], [pad_left, pad_right], [0, 0]]
    # We apply [-1, -2] to the width dimension (index 1).
    paddings = [[0, 0], [-1, -2], [0, 0]]

    print(f"Input shape: {input_tensor.shape}")
    print(f"Attempting explicit padding with values: {paddings}")

    try:
        # Use ksize=1 and strides=1 to isolate the padding behavior
        output = tf.nn.max_pool1d(
            input=input_tensor, 
            ksize=1, 
            strides=1, 
            padding=paddings, 
            data_format='NWC'
        )
        print(f"Success. Output shape: {output.shape}")
    except Exception as e:
        # TensorFlow typically does not support negative padding in explicit lists,
        # unlike the lenient/inconsistent behavior in the PyTorch bug report.
        print(f"Caught Exception: {type(e).__name__}")
        print(f"Message: {e}")

if __name__ == "__main__":
    test_negative_padding_behavior()