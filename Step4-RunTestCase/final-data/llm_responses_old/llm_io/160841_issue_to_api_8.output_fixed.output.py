import tensorflow as tf
import numpy as np

# Enable eager execution to ensure .numpy() method is available on Tensors.
# This is required because the test uses imperative style (asserts on .numpy() results),
# but the environment might default to graph mode (TF1 behavior) where .numpy() is not available.
tf.compat.v1.enable_eager_execution()

def test_tf_compat_v1_math_abs_dtype_precision():
    """
    Test case derived from Issue 160841 (PyTorch phi-2 MacOS garbage output).
    
    The original issue involved incorrect output ("garbage") when using 'auto' dtype
    (often float16 on MacOS) compared to explicitly using 'bf16'. This test verifies
    that tf.compat.v1.math.abs handles these specific dtypes (float16 and bfloat16)
    correctly without producing garbage (NaNs or Infs) or incorrect values.
    """
    
    # Input values that might stress precision limits of float16
    # (e.g., large numbers or small decimals)
    input_vals = [-65000.0, -1.5, 0.0001, 3.14]

    # 1. Test with float16 (The problematic dtype in the original issue)
    # On MacOS/MPS, 'auto' often defaults to float16 which can cause precision issues.
    x_fp16 = tf.constant(input_vals, dtype=tf.float16)
    res_fp16 = tf.compat.v1.math.abs(x_fp16)

    # Assert no NaNs or Infs (Garbage check)
    assert not tf.reduce_any(tf.math.is_nan(res_fp16)).numpy(), "Float16 produced NaNs"
    assert not tf.reduce_any(tf.math.is_inf(res_fp16)).numpy(), "Float16 produced Infs"
    
    # Assert basic correctness (allowing for float16 precision loss)
    expected = np.abs(input_vals)
    assert np.allclose(res_fp16.numpy(), expected, rtol=1e-2, atol=1e-2), "Float16 values incorrect"

    # 2. Test with bfloat16 (The fix mentioned in the original issue)
    # The user reported that changing to bf16 fixed the garbage output.
    x_bf16 = tf.constant(input_vals, dtype=tf.bfloat16)
    res_bf16 = tf.compat.v1.math.abs(x_bf16)

    assert not tf.reduce_any(tf.math.is_nan(res_bf16)).numpy(), "BFloat16 produced NaNs"
    assert not tf.reduce_any(tf.math.is_inf(res_bf16)).numpy(), "BFloat16 produced Infs"
    assert np.allclose(res_bf16.numpy(), expected, rtol=1e-2, atol=1e-2), "BFloat16 values incorrect"

    print("Test passed: tf.compat.v1.math.abs handles float16 and bfloat16 correctly.")

if __name__ == "__main__":
    test_tf_compat_v1_math_abs_dtype_precision()