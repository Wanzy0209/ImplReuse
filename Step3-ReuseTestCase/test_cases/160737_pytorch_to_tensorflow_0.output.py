import tensorflow as tf

def test_triu():
    # Create a 2D input tensor
    x = tf.ones([2, 3], dtype=tf.float32)
    
    # Create a scalar tensor for the offset (k)
    # This mimics the zero-dimensional index tensor in the original bug report
    k_scalar = tf.constant(1)

    try:
        # Attempt to call triu with a scalar tensor for k
        output = tf.experimental.numpy.triu(x, k=k_scalar)
        print(f"triu test succeeds with scalar tensor k. output shape: {output.shape}")
    except Exception as e:
        print(f"triu test fails with scalar tensor k: {e}")

if __name__ == "__main__":
    test_triu()