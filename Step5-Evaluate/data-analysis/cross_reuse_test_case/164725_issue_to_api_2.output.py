import sys

# Handle environment incompatibility errors (e.g., GLIBCXX) during import
try:
    import tensorflow as tf
except ImportError as e:
    print(f"SKIPPED: TensorFlow import failed due to environment incompatibility.")
    print(f"Error: {e}")
    sys.exit(0)

# PyTorch import is kept for context, though not actively used in the translated logic
try:
    import torch
except ImportError:
    pass

def test_executing_eagerly_divergence():
    """
    Test case based on PyTorch Issue 164725.
    
    The original issue highlights a divergence between eager and compiled execution
    when calling .item() on a tensor resulting from masked_select and squeeze.
    
    This test leverages the similar API, tf.compat.v1.executing_eagerly, to verify
    that the execution mode is correctly identified in both contexts. It preserves
    the tensor manipulation logic of the original bug (translated to TensorFlow)
    to ensure the context of the check is relevant to the original failure scenario.
    """

    # Replicate the tensor manipulation logic from the PyTorch issue
    # using TensorFlow equivalents.
    def logic_fn():
        # var_node_4 = torch.full((6,), True, dtype=torch.bool)
        var_node_4 = tf.fill([6], True)
        
        # var_node_3 = torch.reshape(var_node_4, [2, 3])
        var_node_3 = tf.reshape(var_node_4, [2, 3])
        
        # _x_ms = torch.arange(max(1, 1), device=var_node_3.device).to(var_node_3.dtype)
        # max(1, 1) is 1. torch.arange(1) -> [0].
        _x_ms = tf.range(1, dtype=tf.bool)
        
        # _mask_ms = torch.zeros_like(_x_ms, dtype=torch.bool)
        # _mask_ms[:1] = True
        # In TF, we construct the mask directly to avoid tensor mutation
        _mask_ms = tf.constant([True]) 
        
        # var_node_2 = torch.masked_select(_x_ms, _mask_ms)
        var_node_2 = tf.boolean_mask(_x_ms, _mask_ms)
        
        # var_node_1 = torch.squeeze(var_node_2)
        var_node_1 = tf.squeeze(var_node_2)
        
        # The core of the similarity: checking the execution mode.
        # The PyTorch bug manifested because behavior differed between modes.
        # Here we assert that we can correctly detect which mode we are in.
        is_eager = tf.compat.v1.executing_eagerly()
        
        return is_eager, var_node_1

    # 1. Test in Eager Mode
    # The original bug worked correctly in this mode.
    eager_mode, eager_val = logic_fn()
    assert eager_mode == True, "Expected tf.compat.v1.executing_eagerly() to return True in eager mode"
    print(" Eager execution check passed")

    # 2. Test in Graph (Compiled) Mode
    # The original bug (DDE) occurred in this mode.
    compiled_fn = tf.function(logic_fn)
    graph_mode, graph_val = compiled_fn()
    assert graph_mode == False, "Expected tf.compat.v1.executing_eagerly() to return False in graph mode"
    print(" Graph execution check passed")

if __name__ == "__main__":
    test_executing_eagerly_divergence()