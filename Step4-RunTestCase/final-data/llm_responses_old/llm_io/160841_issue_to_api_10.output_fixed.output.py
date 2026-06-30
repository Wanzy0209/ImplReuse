import tensorflow as tf
import numpy as np

# Enable eager execution to allow using Tensors as Python bools in asserts
# and to support .numpy() calls directly, as suggested by the error message.
tf.compat.v1.enable_eager_execution()

def test_tf_keras_backend_abs_precision():
    """
    Test case for tf.keras.backend.abs inspired by Issue 160841.
    
    The original issue involved 'garbage' output when using automatic dtype selection
    (defaulting to float16 on MacOS/MPS) which was resolved by switching to bfloat16.
    
    This test verifies that tf.keras.backend.abs handles these precision-sensitive
    dtypes (float16 and bfloat16) correctly without producing garbage (NaN/Inf).
    """

    # 1. Test with float16 (The problematic dtype in the original issue)
    # We verify that the operation completes without generating NaN or Inf values.
    input_fp16 = tf.constant([-1.5, -2.5, 0.0, 100.0], dtype=tf.float16)
    output_fp16 = tf.keras.backend.abs(input_fp16)

    # Check for garbage output (NaN/Inf)
    assert not tf.reduce_any(tf.math.is_nan(output_fp16)), "NaN detected in float16 output"
    assert not tf.reduce_any(tf.math.is_inf(output_fp16)), "Inf detected in float16 output"
    
    # Verify correctness
    expected_fp16 = tf.constant([1.5, 2.5, 0.0, 100.0], dtype=tf.float16)
    np.testing.assert_allclose(output_fp16.numpy(), expected_fp16.numpy(), rtol=1e-3)

    # 2. Test with bfloat16 (The fix in the original issue)
    input_bf16 = tf.constant([-1.5, -2.5, 0.0, 100.0], dtype=tf.bfloat16)
    output_bf16 = tf.keras.backend.abs(input_bf16)

    # Check for garbage output
    assert not tf.reduce_any(tf.math.is_nan(output_bf16)), "NaN detected in bfloat16 output"
    assert not tf.reduce_any(tf.math.is_inf(output_bf16)), "Inf detected in bfloat16 output"

    # Verify correctness
    expected_bf16 = tf.constant([1.5, 2.5, 0.0, 100.0], dtype=tf.bfloat16)
    np.testing.assert_allclose(output_bf16.numpy(), expected_bf16.numpy(), rtol=1e-2)

    # 3. Test with complex numbers (Specific behavior mentioned in API docs)
    # API docs state: "Given a tensor x of complex numbers, this operation returns 
    # a tensor of type float32 or float64"
    input_complex = tf.constant([3.0 + 4.0j, -5.0 - 12.0j])
    output_complex = tf.keras.backend.abs(input_complex)
    
    assert output_complex.dtype in [tf.float32, tf.float64], "Complex input did not promote to float"
    np.testing.assert_allclose(output_complex.numpy(), [5.0, 13.0])

if __name__ == "__main__":
    test_tf_keras_backend_abs_precision()
    print("Test passed successfully.")