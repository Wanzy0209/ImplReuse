import torch
import tensorflow as tf
import numpy as np

def test_issue_163420_similarity():
    """
    Test case generated based on Issue 163420 (PyTorch) and Similar API tf.nn.dropout.
    
    The original issue involves a divergence between eager and compiled modes when
    using a scalar value extracted from a 0-d tensor (t1.item()) to modify a tensor.
    
    The similar API (tf.nn.dropout) involves scalar-to-tensor conversion and 
    multiplication operations. This test adapts the logic of the similar API
    to accept a 0-d tensor input (mimicking the bug's input structure) and 
    verifies that TensorFlow's graph mode (tf.function) handles the 0-d tensor
    correctly without divergence, similar to the intent of the original bug report.
    """
    
    # Mimicking the structure of the Similar API (tf.nn.dropout internals)
    # but adapted to accept a 0-d tensor for the rate, similar to how the PyTorch
    # bug accepts a 0-d tensor for the fill value.
    def custom_scale_op(x, rate_tensor):
        # x: size=(1, 1), dtype=float32
        # rate_tensor: size=(), dtype=float32 (0-d tensor)
        
        # Pattern from Similar API: convert/cast and math ops
        x_dtype = x.dtype
        keep_prob = 1 - rate_tensor
        scale = 1 / keep_prob
        
        # Ensure scale is a tensor (ops.convert_to_tensor in TF API)
        scale = tf.cast(scale, dtype=x_dtype)
        
        # Pattern from Similar API: gen_math_ops.mul
        ret = tf.multiply(x, scale)
        return ret

    # Inputs mirroring the PyTorch bug report
    # arg0: size=(1, 1)
    arg0 = tf.ones([1, 1], dtype=tf.float32)
    # arg1: size=() (0-d tensor), mimicking t1 in the bug
    arg1 = tf.constant(0.5, dtype=tf.float32)

    # 1. Test Eager Mode
    out_eager = custom_scale_op(arg0, arg1)
    print(f"Eager Output: {out_eager.numpy()}")

    # 2. Test Compiled Mode (tf.function)
    # This corresponds to torch.compile in the original issue
    compiled_op = tf.function(custom_scale_op)
    out_compiled = compiled_op(arg0, arg1)
    print(f"Compiled Output: {out_compiled.numpy()}")

    # 3. Assert Divergence Check
    # The original bug fails here (Triton compilation failed).
    # We assert that TF handles this pattern correctly.
    assert np.allclose(out_eager.numpy(), out_compiled.numpy()), \
        "Divergence detected between eager and compiled modes!"
    
    print('Test Success! ')

if __name__ == '__main__':
    test_issue_163420_similarity()