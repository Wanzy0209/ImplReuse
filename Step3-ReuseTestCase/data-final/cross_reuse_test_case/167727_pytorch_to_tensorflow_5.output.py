import torch
import tensorflow as tf
import numpy as np

def test_iscomplexobj_with_addmm_scenario():
    """
    Adapts the PyTorch addmm bug scenario to test tf.experimental.numpy.iscomplexobj.
    The original bug involved incorrect results for large complex tensors.
    Here, we verify that iscomplexobj correctly identifies the complex nature
    of the tensors involved in similar operations.
    """

    # Helper to mimic torch.addmm(input, mat1, mat2, *, beta=1, alpha=1)
    # result = beta * input + alpha * (mat1 @ mat2)
    def tf_addmm(input_tensor, mat1, mat2, alpha=1.0, beta=1.0):
        matmul_result = tf.linalg.matmul(mat1, mat2)
        return beta * input_tensor + alpha * matmul_result

    # Case 1: Small tensors (Success case in original bug report)
    print("Testing small complex tensors...")
    a = tf.random.uniform((64, 300), dtype=tf.complex64)
    b = tf.random.uniform((64, 100), dtype=tf.complex64)
    c = tf.random.uniform((100, 300), dtype=tf.complex64)
    
    out_small = tf_addmm(a, b, c, alpha=1.0, beta=0.5)
    
    # Verify the API identifies the inputs and result as complex
    assert tf.experimental.numpy.iscomplexobj(a), "Tensor 'a' should be identified as complex"
    assert tf.experimental.numpy.iscomplexobj(b), "Tensor 'b' should be identified as complex"
    assert tf.experimental.numpy.iscomplexobj(c), "Tensor 'c' should be identified as complex"
    assert tf.experimental.numpy.iscomplexobj(out_small), "Result 'out_small' should be identified as complex"

    # Case 2: Large tensors (Failure case in original bug report)
    print("Testing large complex tensors...")
    a = tf.random.uniform((64, 300), dtype=tf.complex64)
    b = tf.random.uniform((64, 10000), dtype=tf.complex64)
    c = tf.random.uniform((10000, 300), dtype=tf.complex64)
    
    out_large = tf_addmm(a, b, c, alpha=1.0, beta=0.5)
    
    # Verify the API identifies the inputs and result as complex
    assert tf.experimental.numpy.iscomplexobj(a), "Large tensor 'a' should be identified as complex"
    assert tf.experimental.numpy.iscomplexobj(b), "Large tensor 'b' should be identified as complex"
    assert tf.experimental.numpy.iscomplexobj(c), "Large tensor 'c' should be identified as complex"
    assert tf.experimental.numpy.iscomplexobj(out_large), "Result 'out_large' should be identified as complex"

    print("All assertions passed.")

if __name__ == "__main__":
    test_iscomplexobj_with_addmm_scenario()