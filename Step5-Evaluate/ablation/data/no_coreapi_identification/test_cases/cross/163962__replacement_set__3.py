import torch
import tensorflow as tf
import tensorly as tl
from tensorly.decomposition import parafac
import pytest

def test_parafac_mps_with_replacements():
    """
    Test case based on Issue 163962, adapted with specific API replacements.
    Original bug: Internal assert failed when using Tensorly on MPS.
    Similar API: tf.errors.OperatorNotAllowedInGraphError.
    Replacements applied:
    - torch.ones -> torch.linspace
    - x.detach -> tf.autograph.trace
    - tl.solve -> tf.autograph.trace
    - tl.reshape -> tf.autograph.trace
    """
    # Setup Tensorly backend
    tl.set_backend("pytorch")

    # Replacement: torch.ones -> torch.linspace
    # Original: x = torch.ones(12,3,12).to("mps")
    # torch.linspace creates a 1D tensor, so we reshape to match the original dimensions.
    x = torch.linspace(0, 1, 12 * 3 * 12).reshape(12, 3, 12).to("mps")
    print(f"Input shape: {x.shape}")

    # Replacements: tl.solve -> tf.autograph.trace, tl.reshape -> tf.autograph.trace
    # We replace the tensorly backend functions with their traced versions.
    # Note: We capture the original functions to restore them later, though in a test
    # context, isolation is usually handled by the test runner.
    original_solve = tl.solve
    original_reshape = tl.reshape

    # Applying the replacements
    tl.solve = tf.autograph.trace(original_solve)
    tl.reshape = tf.autograph.trace(original_reshape)

    try:
        # Replacement: x.detach -> tf.autograph.trace
        # Original: parafac(x.detach(), 12, init="random", tol=1e-6)
        # We interpret the replacement as tracing the tensor access/detachment logic.
        # Since x is a tensor, we wrap it in a lambda to satisfy tf.autograph.trace's signature.
        x_input = tf.autograph.trace(lambda: x)()

        # Execute the decomposition
        # We expect this to potentially raise an error similar to the Similar API
        # (tf.errors.OperatorNotAllowedInGraphError) because we are forcing TF tracing
        # on PyTorch operations, which are not supported in TF graphs.
        with pytest.raises((tf.errors.OperatorNotAllowedInGraphError, RuntimeError, AttributeError)):
            weights, factors = parafac(x_input, 12, init="random", tol=1e-6)
            
            # If no error is raised (e.g. if autograph ignores unknown ops), we check the output
            a, m, b = factors
            assert a is not None
            assert m is not None
            assert b is not None

    finally:
        # Restore original functions
        tl.solve = original_solve
        tl.reshape = original_reshape

if __name__ == "__main__":
    test_parafac_mps_with_replacements()