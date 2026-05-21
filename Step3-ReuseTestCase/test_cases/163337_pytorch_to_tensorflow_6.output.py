import tensorflow as tf
import numpy as np

def test_tf_compat_v1_multinomial():
    """
    Test case for tf.compat.v1.multinomial.
    
    Note: The original bug report (Issue 163337) describes a compilation error 
    when building a PyTorch C++ extension using rocWMMA headers. 
    Since tf.compat.v1.multinomial is a pre-built TensorFlow operation and does not 
    support JIT compilation of arbitrary C++ sources via its Python API, the specific 
    compilation error cannot be directly reproduced here. 
    
    This test verifies the runtime behavior of the identified similar API 
    (tf.compat.v1.multinomial) to ensure it functions correctly on the system.
    """
    
    # Enable eager execution for testing
    if not tf.executing_eagerly():
        tf.compat.v1.enable_eager_execution()

    # Define logits (2-D Tensor with shape [batch_size, num_classes])
    # Using float32, which corresponds to the 'float' type mentioned in the bug report
    logits = tf.constant([[1.0, 1.0, 1.0, 1.0], 
                          [10.0, 1.0, 1.0, 1.0]], dtype=tf.float32)
    
    num_samples = 5
    seed = 42
    
    # Call the similar API
    try:
        samples = tf.compat.v1.multinomial(
            logits, 
            num_samples, 
            seed=seed, 
            output_dtype=tf.int64
        )
        
        # Verify output shape
        expected_shape = (2, 5)
        assert samples.shape == expected_shape, \
            f"Expected shape {expected_shape}, but got {samples.shape}"
            
        # Verify output dtype
        assert samples.dtype == tf.int64, \
            f"Expected dtype tf.int64, but got {samples.dtype}"
            
        # Verify values are within valid range [0, num_classes)
        num_classes = logits.shape[1]
        assert tf.reduce_all(samples >= 0), "Samples contain negative values"
        assert tf.reduce_all(samples < num_classes), \
            f"Samples contain values >= num_classes ({num_classes})"
            
        print("Test passed: tf.compat.v1.multinomial executed successfully.")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise

if __name__ == "__main__":
    test_tf_compat_v1_multinomial()