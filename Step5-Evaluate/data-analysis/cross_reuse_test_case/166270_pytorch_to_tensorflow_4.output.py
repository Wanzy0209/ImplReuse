import torch
import tensorflow as tf
import numpy as np

# Set seed for reproducibility
tf.random.set_seed(1061983224)

# Define the program logic using the target API: tf.keras.ops.tril
# We adapt the shape manipulation logic (chunk -> squeeze -> stack -> reshape)
# to fit around the tril operation.
def program_logic(input_tensor):
    # Apply the target API
    # tril requires rank >= 2, so we ensure input is at least 2D
    x = tf.keras.ops.tril(input_tensor)
    
    # Mimic the original shape manipulation sequence:
    # Original: chunk -> squeeze -> stack -> reshape
    
    # 1. Slice (mimicking chunk)
    # Original: var_node_3 = torch.chunk(var_node_4, 4, dim=0)[0]
    # We take the first row of the matrix
    var_node_3 = x[0] 
    
    # 2. Squeeze
    # Original: var_node_2 = torch.squeeze(var_node_3) -> size ()
    # In the original, var_node_3 was size (1,), so squeeze made it ().
    # We reshape the slice to (1,) to match the original intermediate state.
    var_node_3_reshaped = tf.reshape(var_node_3, [1]) # (1,)
    var_node_2 = tf.squeeze(var_node_3_reshaped) # ()
    
    # 3. Stack
    # Original: var_node_1 = torch.stack([var_node_2], dim=0) -> (1,)
    var_node_1 = tf.stack([var_node_2], axis=0)
    
    # 4. Reshape
    # Original: var_node_0 = torch.reshape(var_node_1, [1]) -> (1,)
    var_node_0 = tf.reshape(var_node_1, [1])
    
    return var_node_0

# 1. Eager Execution
print("Running eager execution...")
# Create input: (4, 4) bool tensor to satisfy tril requirements
# Original used bool, we stick to it.
arg_0 = tf.constant(np.random.randint(0, 2, (4, 4)), dtype=tf.bool)

result_eager = program_logic(arg_0)
print(' eager success')

# 2. Compiled Execution
print("Running compiled execution...")
# Use tf.function to mimic torch.compile
compiled_program = tf.function(program_logic, jit_compile=True)

try:
    result_compiled = compiled_program(arg_0)
    print(' compile success')
    
    # Verify consistency
    if tf.reduce_all(tf.equal(result_eager, result_compiled)).numpy():
        print(' results match')
    else:
        print(' results diverge')
        
except Exception as e:
    print(f' compile failed: {e}')