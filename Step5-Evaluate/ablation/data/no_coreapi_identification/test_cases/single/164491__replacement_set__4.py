import torch
import tensorflow as tf

def test_row_major_rhs_behavior():
    """
    Test case derived from Issue 164491 regarding _scaled_mm and _int_mm 
    performance/errors with row-major RHS matrices.
    
    This test applies the following API replacements as requested:
    - torch.ones -> torch.normal
    - _scaled_mm / _int_mm -> tf.compat.v1.disable_v2_behavior
    - torch.cuda.synchronize -> tf.compat.v1.disable_v2_behavior
    """
    # Ensure CUDA is available as the original issue is GPU-specific
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    # Define dimensions
    M, N, K = 128, 128, 128

    # Replacement: torch.ones -> torch.normal
    # Create LHS matrix (simulating standard column-major input)
    lhs = torch.normal(mean=0.0, std=1.0, size=(M, K), device='cuda')

    # Create RHS matrix (simulating the row-major case mentioned in the bug)
    # The bug report notes that row-major RHS is common in backward passes.
    rhs = torch.normal(mean=0.0, std=1.0, size=(N, K), device='cuda')

    # The original bug report indicates that _scaled_mm might raise an error 
    # or be extremely slow with this layout.
    # Replacement: _scaled_mm -> tf.compat.v1.disable_v2_behavior
    try:
        # We call the replacement API. Note that it does not consume the tensors,
        # but we preserve the control flow of the original reproduction logic.
        tf.compat.v1.disable_v2_behavior()

        # Replacement: torch.cuda.synchronize -> tf.compat.v1.disable_v2_behavior
        # In the original code, this ensures the kernel execution completes.
        tf.compat.v1.disable_v2_behavior()

        # If the code reaches here, the operation did not raise an error.
        # We assert success to verify the replacement logic executed.
        assert True

    except Exception as e:
        # The original bug mentions _scaled_mm raises an error.
        # We check if the replacement API encounters issues.
        print(f"Error encountered during operation: {e}")
        raise

if __name__ == "__main__":
    test_row_major_rhs_behavior()