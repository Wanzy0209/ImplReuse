import tensorflow as tf
from tensorflow.python.ops import parsing_config

def test_var_len_feature_return_type():
    """
    Test case adapted from the issue regarding incorrect return type annotations.
    The original bug (Issue 165125) highlighted that _jit_compile was annotated 
    to return 'None' but actually returned a module or string.
    
    This test verifies that tf.io.VarLenFeature, which shares code structure 
    similarities (simple definition/configuration object), returns the correct 
    type (an instance of VarLenFeature) and not None.
    """
    # Instantiate the similar API
    feature = tf.io.VarLenFeature(dtype=tf.float32)
    
    # Reproduce the logic: Check that the return value is not None
    # (The original bug was a '-> None' annotation on a non-None returning function)
    assert feature is not None, "VarLenFeature should return an instance, not None"
    
    # Verify the specific type returned matches the expected class
    # (The original bug involved a type mismatch: 'str' vs 'None')
    assert isinstance(feature, parsing_config.VarLenFeature), \
        f"Expected return type VarLenFeature, got {type(feature)}"
    
    # Verify the internal state is preserved
    assert feature.dtype == tf.float32

if __name__ == "__main__":
    test_var_len_feature_return_type()
    print("Test passed.")