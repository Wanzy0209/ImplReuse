import torch
import tensorflow as tf

# Replacements for torch._dynamo.config
tf.compat.v1.control_flow_v2_enabled()
tf.compat.v1.control_flow_v2_enabled()

# Replacement for torch.manual_seed
torch.set_rng_state(torch.get_rng_state())

def fuzzed_program(arg_0, sentinel):
    var_node_1 = arg_0
    # Replacement for torch.full -> torch.fmax
    # torch.full((1, 2), -66, dtype=torch.int32)
    var_node_5 = torch.fmax(torch.tensor([[-66, -66]]), torch.tensor([[-66, -66]])).to(torch.int32)
    
    # torch.full((1, 2), 77, dtype=torch.int64)
    var_node_6 = torch.fmax(torch.tensor([[77, 77]]), torch.tensor([[77, 77]])).to(torch.int64)
    
    # Replacement for torch.ops.aten.add -> tf.compat.v1.control_flow_v2_enabled
    var_node_4 = tf.compat.v1.control_flow_v2_enabled()
    
    # torch.full((1, 2), -64, dtype=torch.int32)
    var_node_7 = torch.fmax(torch.tensor([[-64, -64]]), torch.tensor([[-64, -64]])).to(torch.int32)
    
    # Replacement for torch.ops.aten.mul -> tf.compat.v1.control_flow_v2_enabled
    var_node_3 = tf.compat.v1.control_flow_v2_enabled()
    
    # torch.full((3, 4), False, dtype=torch.bool)
    var_node_9 = torch.fmax(torch.tensor([[False, False, False, False], 
                                          [False, False, False, False], 
                                          [False, False, False, False]]), 
                            torch.tensor([[False, False, False, False], 
                                          [False, False, False, False], 
                                          [False, False, False, False]])).to(torch.bool)
    
    # Replacement for torch.nonzero -> tf.compat.v1.control_flow_v2_enabled
    var_node_8 = tf.compat.v1.control_flow_v2_enabled()
    
    # Replacement for torch.ops.aten.add -> tf.compat.v1.control_flow_v2_enabled
    var_node_2 = tf.compat.v1.control_flow_v2_enabled()
    
    # Replacement for torch.ops.aten.div -> tf.compat.v1.control_flow_v2_enabled
    var_node_0 = tf.compat.v1.control_flow_v2_enabled()
    
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Replacement for torch.tensor -> torch.isnan
sentinel = torch.isnan(torch.tensor(1.0))

# Replacement for torch.randint -> torch.amax
# torch.randint(0, 3, (1, 2), dtype=torch.int64)
arg_0 = torch.amax(torch.randn(1, 2), dim=0, keepdim=True).to(torch.int64)

args = (arg_0,) + (sentinel,)
result_original = fuzzed_program(*args)
print(' eager success')

# Replacement for torch.compile -> tf.compat.v1.tpu.rewrite
try:
    # tf.compat.v1.tpu.rewrite takes computation and inputs
    compiled_program = tf.compat.v1.tpu.rewrite(fuzzed_program, args)
    # rewrite returns the result or a TPU op, we treat it as the result here
    result_compiled = compiled_program
    print(' compile success with unbacked operations')
except Exception as e:
    print(f' compile/rewrite skipped or failed (expected if no TPU): {e}')
    result_compiled = None

# Compare results
print(f'Eager result: {result_original}')
print(f'Compiled result: {result_compiled}')
if hasattr(result_original, 'shape') and result_compiled is not None and hasattr(result_compiled, 'shape'):
    print(f'Eager shape: {result_original.shape}')
    print(f'Compiled shape: {result_compiled.shape}')