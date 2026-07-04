import sys

try:
    import torch
    import tensorflow as tf
except ImportError as e:
    # Handle the specific environment error regarding GLIBCXX
    if "GLIBCXX" in str(e) or "libstdc++" in str(e):
        print("SKIP: Test skipped due to missing system dependencies (GLIBCXX_3.4.29).")
        print(f"Error details: {e}")
        sys.exit(0)
    else:
        raise

def fuzzed_program():
    """
    Reproduces the tensor manipulation logic from the PyTorch issue
    to test the behavior of tf.executing_eagerly in different contexts.
    """
    # var_node_4 = torch.full((6,), True, dtype=torch.bool)
    var_node_4 = tf.fill([6], True)
    
    # var_node_3 = torch.reshape(var_node_4, [2, 3])
    var_node_3 = tf.reshape(var_node_4, [2, 3])
    
    # var_node_5 = torch.full((2, 3), False, dtype=torch.bool)
    # Kept for structural similarity, though not strictly used in the final chain
    var_node_5 = tf.fill([2, 3], False)
    
    # _x_ms = torch.arange(max(1, 1), device=var_node_3.device).to(var_node_3.dtype)
    # torch.arange(1) -> [0], to(bool) -> False
    _x_ms = tf.range(1, dtype=tf.bool)
    
    # _mask_ms = torch.zeros_like(_x_ms, dtype=torch.bool)
    # _mask_ms[:1] = True
    # Constructing the mask: [True]
    _mask_ms = tf.concat([tf.constant([True]), tf.zeros_like(_x_ms)[1:]], axis=0)
    
    # var_node_2 = torch.masked_select(_x_ms, _mask_ms)
    # Selects the False value because mask is True
    var_node_2 = tf.boolean_mask(_x_ms, _mask_ms)
    
    # var_node_1 = torch.squeeze(var_node_2)
    # Squeezes the single element to a scalar tensor
    var_node_1 = tf.squeeze(var_node_2)
    
    # Check execution mode using the similar API
    is_eager = tf.executing_eagerly()
    
    # var_node_0 = var_node_1.item()
    # The original bug involved .item() behaving differently in eager vs compile.
    # Here we use tf.executing_eagerly to handle the scalar extraction appropriately,
    # mimicking the need to distinguish execution modes.
    if is_eager:
        # In eager mode, we can extract the python scalar immediately
        scalar_val = var_node_1.numpy().item()
    else:
        # In graph mode (tf.function), we return the tensor
        scalar_val = var_node_1
        
    return is_eager, scalar_val

# Test Eager Execution
print("Testing Eager Execution...")
result_eager_mode, result_eager_val = fuzzed_program()
assert result_eager_mode == True, "Expected eager execution to be True"
assert result_eager_val == False, "Expected scalar value to be False (bool)"
print(' eager success')

# Test Compiled Execution (tf.function)
print("Testing Compiled Execution...")
compiled_program = tf.function(fuzzed_program)
result_compiled_mode, result_compiled_val = compiled_program()

# tf.executing_eagerly should return False inside tf.function
assert result_compiled_mode == False, "Expected eager execution to be False inside tf.function"
# In graph mode, result_compiled_val is a Tensor, not a python scalar
assert isinstance(result_compiled_val, tf.Tensor), "Expected Tensor result in graph mode"
print(' compile success')