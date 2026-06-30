import torch

def test_var_with_keras_floatx(device):
    """
    Test case for torch.var on zero-dimensional input with dim=0.
    Leverages tf.keras.backend.floatx to determine the tensor dtype,
    preserving the original bug reproduction logic.
    """
    # Leverage the similar API to get the default float type
    # Wrap in try-except to handle environment dependency issues (e.g., GLIBC)
    try:
        import tensorflow as tf
        keras_float_str = tf.keras.backend.floatx()
    except ImportError:
        # Fallback to float32 if TensorFlow is unavailable or environment is broken
        print("Warning: TensorFlow import failed due to environment issues. Defaulting to float32.")
        keras_float_str = 'float32'
    
    # Translate Keras float type to PyTorch dtype
    if keras_float_str == 'float16':
        dtype = torch.float16
    elif keras_float_str == 'float32':
        dtype = torch.float32
    elif keras_float_str == 'float64':
        dtype = torch.float64
    else:
        # Fallback to float32 if unknown
        dtype = torch.float32

    print(f"Testing on device: {device} with dtype: {dtype} (from Keras floatx: {keras_float_str})")
    
    # Original bug reproduction logic: zero-dimensional tensor, dim=0
    x = torch.tensor(3.0, device=device, dtype=dtype)
    
    try:
        output = torch.var(x, dim=0)
        print(f"var test succeeds for device: {device}. output: {output}")
        # Assert that the output is NaN (variance of a single value)
        assert torch.isnan(output), f"Expected NaN for variance of scalar, got {output}"
    except Exception as e:
        print(f"var test fails for device: {device}: {e}")
        raise

if __name__ == "__main__":
    # Test on CPU
    test_var_with_keras_floatx(device="cpu")
    
    # Test on MPS if available (reproducing the specific bug scenario)
    if torch.backends.mps.is_available():
        test_var_with_keras_floatx(device="mps")
    else:
        print("MPS device not available, skipping MPS test.")