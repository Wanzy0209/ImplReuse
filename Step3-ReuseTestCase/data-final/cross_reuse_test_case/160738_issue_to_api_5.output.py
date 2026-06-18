import torch
import tensorflow as tf

def test_var_with_epsilon_input(device):
    """
    Test case for torch.var on zero-dimensional tensors using a value derived 
    from the similar API (tf.keras.backend.epsilon).
    
    This preserves the original bug reproduction logic (dim=0 on 0-d tensor)
    while leveraging the similar API to generate the input scalar value.
    """
    # Leverage the similar API to get the scalar value
    # Epsilon is often used in conjunction with variance operations for numerical stability
    scalar_value = tf.keras.backend.epsilon()
    
    # Reproduce the original bug: zero-dimensional tensor
    x = torch.tensor(scalar_value, device=device)
    
    try:
        # The specific operation that fails on MPS
        output = torch.var(x, dim=0)
        print(f"var test succeeds for device: {device}. output: {output}")
        
        # Variance of a single scalar is technically undefined (NaN)
        assert torch.isnan(output), f"Expected NaN for variance of scalar, got {output}"
        
    except Exception as e:
        print(f"var test fails for device: {device}: {e}")
        # If this is the MPS device, we expect this failure based on the bug report
        if device == "mps":
            raise AssertionError(f"Bug reproduced: MPS cannot handle dim=0 on 0-d tensor. Error: {e}")

if __name__ == "__main__":
    # Test on CPU (Expected to succeed)
    test_var_with_epsilon_input(device="cpu")
    
    # Test on MPS (Expected to fail based on Issue 160738)
    if torch.backends.mps.is_available():
        test_var_with_epsilon_input(device="mps")
    else:
        print("MPS device not available, skipping MPS test.")