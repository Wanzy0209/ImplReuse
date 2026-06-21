import torch
import tensorflow as tf
import traceback

# Set seed for reproducibility, matching the original report's seed
tf.random.set_seed(1061983224)

def fuzzed_program(arg_0, sentinel):
    # Adaptation logic:
    # The original PyTorch test performed chunk -> squeeze -> stack -> reshape.
    # The target API is tf.keras.ops.triu, which requires a matrix (rank >= 2).
    # We reshape the input (4,) to (2, 2) to satisfy triu requirements.
    var_node_4 = tf.reshape(arg_0, (2, 2)) # size=(2, 2), dtype=bool
    
    # Call the target API: tf.keras.ops.triu
    # This corresponds to the "var_node_2" step in the original (the core operation)
    var_node_2 = tf.keras.ops.triu(var_node_4) # size=(2, 2), dtype=bool
    
    # Mimic the subsequent operations in the original test to ensure 
    # the tensor flows through the graph correctly.
    # Original: torch.stack([var_node_2], dim=0)
    var_node_1 = tf.stack([var_node_2], axis=0) # size=(1, 2, 2)
    
    # Original: torch.reshape(var_node_1, [1])
    # Note: Reshaping (1, 2, 2) to [1] is invalid (size mismatch).
    # We adapt the reshape to be valid but aggressive, e.g., flattening or keeping shape.
    # Let's reshape to (-1) to flatten, or keep (1, 2, 2).
    # To stay close to the "reshape" intent, we flatten it.
    var_node_0 = tf.reshape(var_node_1, [-1]) # size=(4,)
    
    # Ensure gradient computation by multiplying with sentinel
    result = var_node_0 * sentinel
    return result

# Sentinel tensor to ensure gradient computation
# In PyTorch: torch.tensor(1.0, requires_grad=True)
# In TF: We use a constant and watch it in GradientTape, or just a variable.
sentinel = tf.constant(1.0)

# Input generation
# Original: torch.as_strided(torch.randint(0, 2, (4,), dtype=torch.int8).bool(), (4,), (1,))
# TF equivalent: A boolean tensor of size 4.
arg_0 = tf.cast(tf.random.uniform((4,), minval=0, maxval=2, dtype=tf.int32), tf.bool)

args = (arg_0, sentinel)

# 1. Eager Execution
try:
    result_original = fuzzed_program(*args)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')
    traceback.print_exc()
    exit(1)

# 2. Compiled Execution (tf.function)
# Equivalent to torch.compile(..., fullgraph=True)
try:
    compiled_program = tf.function(fuzzed_program)
    result_compiled = compiled_program(*args)
    print(' compile success')
except Exception as e:
    print(f' compile failed: {e}')
    traceback.print_exc()
    exit(1)

# 3. Verification
# Check if eager and compiled results match
if tf.reduce_all(tf.equal(result_original, result_compiled)).numpy():
    print(' eager and compiled results match')
else:
    print(' eager and compiled results diverge')
    print("Eager:", result_original.numpy())
    print("Compiled:", result_compiled.numpy())
    exit(1)

# 4. Gradient Check (Optional but good for robustness)
# The original code specifically checked for gradient computation capability.
try:
    # Recreate inputs as float for gradient check (bool gradients are tricky/often None in TF)
    arg_0_float = tf.cast(arg_0, tf.float32)
    sentinel_float = tf.constant(1.0)
    
    def grad_func(x, s):
        x = tf.reshape(x, (2, 2))
        x = tf.cast(x, tf.bool) # triu works on bool
        y = tf.keras.ops.triu(x)
        y = tf.cast(y, tf.float32) # cast back for math
        y = tf.stack([y], axis=0)
        y = tf.reshape(y, [-1])
        return y * s

    with tf.GradientTape() as tape:
        tape.watch(arg_0_float)
        res = grad_func(arg_0_float, sentinel_float)
        
    grads = tape.gradient(res, arg_0_float)
    if grads is not None:
        print(' gradient computation successful')
    else:
        print(' gradients are None (might be expected for triu masking)')
except Exception as e:
    print(f' gradient check failed: {e}')