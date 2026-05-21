import torch
import tensorflow as tf

def test_tf_exp_divergence():
    """
    Test case for tf.compat.v1.math.exp inspired by PyTorch Issue 164725.
    The original issue involves a divergence between eager and compiled modes
    when extracting a scalar (.item()) after a series of tensor manipulations.
    This test adapts the logic to use tf.compat.v1.math.exp and checks for
    eager vs. tf.function divergence.
    """
    
    # Replicate the tensor manipulation logic from the PyTorch bug report
    def fuzzed_program():
        # var_node_4 = torch.full((6,), True, dtype=torch.bool)
        var_node_4 = tf.fill([6], True) # dtype=bool implied by fill value
        
        # var_node_3 = torch.reshape(var_node_4, [2, 3])
        var_node_3 = tf.reshape(var_node_4, [2, 3])
        
        # var_node_5 = torch.full((2, 3), False, dtype=torch.bool)
        var_node_5 = tf.fill([2, 3], False)
        
        # _x_ms = torch.arange(max(1, 1), device=var_node_3.device).to(var_node_3.dtype)
        _x_ms = tf.cast(tf.range(max(1, 1)), var_node_3.dtype)
        
        # _mask_ms = torch.zeros_like(_x_ms, dtype=torch.bool)
        _mask_ms = tf.zeros_like(_x_ms, dtype=tf.bool)
        
        # _mask_ms[:1] = True
        # In TensorFlow, we use scatter_nd_update to modify specific indices
        _mask_ms = tf.tensor_scatter_nd_update(_mask_ms, [[0]], [True])
        
        # var_node_2 = torch.masked_select(_x_ms, _mask_ms)
        var_node_2 = tf.boolean_mask(_x_ms, _mask_ms)
        
        # var_node_1 = torch.squeeze(var_node_2)
        var_node_1 = tf.squeeze(var_node_2)
        
        # Original PyTorch code calls .item() here: var_node_0 = var_node_1.item()
        # We leverage the similar API: tf.compat.v1.math.exp
        # Note: exp requires float/complex input, so we cast the bool result to float.
        var_node_0 = tf.compat.v1.math.exp(tf.cast(var_node_1, tf.float32))
        
        # Sentinel logic to ensure gradient computation context (mimicking the original)
        sentinel = tf.constant(1.0)
        result = var_node_0 * sentinel
        
        return result

    # 1. Run in Eager Mode
    result_eager = fuzzed_program()
    print(f" Eager result: {result_eager.numpy()}")

    # 2. Run in Compiled Mode (tf.function equivalent to torch.compile)
    compiled_program = tf.function(fuzzed_program)
    result_compiled = compiled_program()
    print(f" Compiled result: {result_compiled.numpy()}")

    # 3. Check for Divergence
    # The original bug reported a DDE (Dynamic Divergence Error) or incorrect behavior.
    # We assert that both results are equal.
    if not tf.reduce_all(tf.equal(result_eager, result_compiled)).numpy():
        raise AssertionError("Divergence detected between eager and compiled execution!")
    
    print(" Test passed: No divergence detected.")

if __name__ == "__main__":
    test_tf_exp_divergence()