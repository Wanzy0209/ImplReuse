import torch
import pytest

# Handle environment/dependency issues (e.g., GLIBCXX version mismatch)
try:
    import tensorflow as tf
except ImportError as e:
    pytest.skip(f"Skipping test because TensorFlow could not be imported: {e}", allow_module_level=True)

# Issue 164924: torch.isin fails for scalar test_elements in compile.
# The bug is triggered by passing a 0-dimensional tensor (scalar) to an operation.
# This test case applies the same logic to the similar API tf.keras.ops.diagonal,
# checking its behavior when the input tensor is a scalar.

def test_diagonal_with_scalar_input():
    """
    Test tf.keras.ops.diagonal with a scalar input.
    Mirrors the logic of Issue 164924 where torch.isin failed with a scalar test_elements.
    """
    # Setup: Create a scalar tensor (0-d), analogous to 'self.y' in the original bug.
    # Original: self.y = torch.randint(-50, 50, (), dtype=torch.int64, device=device)
    scalar_input = tf.constant(42, dtype=tf.int64)

    # The original bug resulted in a failure/crash in the compiled backend.
    # We check if tf.keras.ops.diagonal handles the scalar input gracefully.
    # Based on the implementation snippet, it checks rank and calls matrix_diag_part.
    # matrix_diag_part typically requires rank >= 2, so we expect an error.
    with pytest.raises((tf.errors.InvalidArgumentError, ValueError)):
        tf.keras.ops.diagonal(scalar_input)

def test_diagonal_with_compiled_scalar_input():
    """
    Test tf.keras.ops.diagonal with a scalar input inside tf.function (compiled).
    Mirrors the 'torch.compile' aspect of the original bug.
    """
    scalar_input = tf.constant(42, dtype=tf.int64)

    @tf.function
    def compiled_diagonal(x):
        return tf.keras.ops.diagonal(x)

    # Expect the same error behavior in compiled mode
    with pytest.raises((tf.errors.InvalidArgumentError, ValueError)):
        compiled_diagonal(scalar_input)