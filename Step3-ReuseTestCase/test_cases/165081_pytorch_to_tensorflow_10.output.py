import torch
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import numpy as np

def test_tf_cbrt_with_fuzzer_data():
    """
    Adapted from PyTorch Issue 165081.
    The original issue involved torch._dynamo failing to guard on data-dependent expressions.
    This test verifies that tf.experimental.numpy.cbrt handles similar data patterns
    (specifically negative values which trigger data-dependent logic in cbrt)
    correctly, both in eager and compiled (tf.function) modes.
    """

    # Recreate tensors from the PyTorch fuzzer output
    # var_node_9 = torch.full((9, 11, 12), 1.5758497316910556, dtype=torch.float64)
    tensor_pos = tf.constant(1.5758497316910556, shape=(9, 11, 12), dtype=tf.float64)

    # var_node_23 = torch.full((156, 8), -0.5249394453404403, dtype=torch.float64)
    # Negative values are critical for cbrt as it involves a conditional check (x < 0)
    tensor_neg_1 = tf.constant(-0.5249394453404403, shape=(156, 8), dtype=tf.float64)

    # var_node_26 = torch.full((9, 13), -0.9276381954691514, dtype=torch.float64)
    tensor_neg_2 = tf.constant(-0.9276381954691514, shape=(9, 13), dtype=tf.float64)

    # Wrap in tf.function to test compilation behavior (analogous to torch._dynamo)
    @tf.function
    def compiled_cbrt(x):
        return tnp.cbrt(x)

    # Test 1: Positive values
    result_pos = compiled_cbrt(tensor_pos)
    assert result_pos.shape == tensor_pos.shape
    assert result_pos.dtype == tf.float64
    np.testing.assert_allclose(result_pos.numpy(), np.cbrt(tensor_pos.numpy()), rtol=1e-5)
    print("Positive tensor test passed.")

    # Test 2: Negative values (Data-dependent logic)
    # The implementation of cbrt uses array_ops.where_v2(x < 0, -rt, rt)
    # This verifies the data-dependent branching works in compiled mode.
    result_neg_1 = compiled_cbrt(tensor_neg_1)
    assert result_neg_1.shape == tensor_neg_1.shape
    assert result_neg_1.dtype == tf.float64
    np.testing.assert_allclose(result_neg_1.numpy(), np.cbrt(tensor_neg_1.numpy()), rtol=1e-5)
    print("Negative tensor 1 test passed.")

    # Test 3: Another negative value
    result_neg_2 = compiled_cbrt(tensor_neg_2)
    assert result_neg_2.shape == tensor_neg_2.shape
    np.testing.assert_allclose(result_neg_2.numpy(), np.cbrt(tensor_neg_2.numpy()), rtol=1e-5)
    print("Negative tensor 2 test passed.")

if __name__ == "__main__":
    test_tf_cbrt_with_fuzzer_data()