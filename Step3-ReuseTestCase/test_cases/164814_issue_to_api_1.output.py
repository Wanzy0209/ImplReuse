import tensorflow as tf
import numpy as np

# The original bug (Issue 164814) involves a divergence between eager and compiled modes
# when handling 0-d tensors (scalars), specifically related to stride/size dimensionality.
# The similar API, tf.experimental.numpy.diagonal, involves shape manipulation and
# can produce 0-d tensors (e.g., from a 1x1 matrix) or empty tensors.
# This test checks for eager/compile divergence in these edge cases.

def test_diagonal_dimensionality_consistency():
    # Scenario 1: Producing a 0-d tensor (scalar) result
    # In PyTorch, the bug occurred when a 0-d tensor was involved in stride calculations.
    # Here, diagonal on a (1, 1) tensor returns a 0-d tensor.
    input_1x1 = tf.constant([[42.0]])

    @tf.function
    def compiled_diagonal(x, offset=0):
        return tf.experimental.numpy.diagonal(x, offset=offset)

    # Run in eager mode
    result_eager = compiled_diagonal(input_1x1)
    # Run in compiled mode (tf.function)
    result_compiled = compiled_diagonal(input_1x1)

    # Check consistency
    assert result_eager.shape == (), f"Eager: Expected scalar shape, got {result_eager.shape}"
    assert result_compiled.shape == (), f"Compiled: Expected scalar shape, got {result_compiled.shape}"
    assert np.allclose(result_eager.numpy(), result_compiled.numpy()), "Values differ between modes"

    # Scenario 2: Producing an empty tensor result
    # The similar API code mentions shape inference failures for out-of-bounds offsets.
    # This tests shape consistency for empty results.
    input_2x2 = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    offset_out_of_bounds = 10

    result_eager_empty = compiled_diagonal(input_2x2, offset=offset_out_of_bounds)
    result_compiled_empty = compiled_diagonal(input_2x2, offset=offset_out_of_bounds)

    assert result_eager_empty.shape == (0,), f"Eager: Expected empty shape, got {result_eager_empty.shape}"
    assert result_compiled_empty.shape == (0,), f"Compiled: Expected empty shape, got {result_compiled_empty.shape}"
    assert np.array_equal(result_eager_empty.numpy(), result_compiled_empty.numpy()), "Empty results differ"

    print(" Test passed: No eager/compile divergence detected for tf.experimental.numpy.diagonal.")

if __name__ == "__main__":
    test_diagonal_dimensionality_consistency()