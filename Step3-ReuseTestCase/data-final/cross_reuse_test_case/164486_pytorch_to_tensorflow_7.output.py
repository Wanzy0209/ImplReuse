import torch
import tensorflow as tf
import numpy as np

# Set seed for reproducibility
tf.random.set_seed(238)

def fuzzed_program(sp_input_1, sp_input_2):
    # Original API: torch.div
    # Similar API: tf.sparse.concat
    # We adapt the logic to concatenate two sparse tensors.
    # The original test involved scalar tensors (0-D). 
    # tf.sparse.concat requires rank >= 1, so we use 1-D tensors.
    return tf.sparse.concat([sp_input_1, sp_input_2], axis=0)

# Prepare inputs
# Original used dtype=torch.int16. We use tf.int16 to match the type constraint.
dtype = tf.int16

# Input 1: Analogous to var_node_1 (scalar value 1)
# Shape (1,) to mimic the scalar nature in a concat-compatible way
indices_1 = [[0]]
values_1 = [1]
shape_1 = [1]
sp_input_1 = tf.sparse.SparseTensor(indices_1, values_1, shape_1)

# Input 2: Analogous to var_node_4 (scalar value 3)
# Shape (1,)
indices_2 = [[0]]
values_2 = [3]
shape_2 = [1]
sp_input_2 = tf.sparse.SparseTensor(indices_2, values_2, shape_2)

# Eager Execution
print('Running Eager...')
try:
    out_eager = fuzzed_program(sp_input_1, sp_input_2)
    # Convert to dense for easier comparison
    dense_eager = tf.sparse.to_dense(out_eager)
    print('Eager Success! ')
except Exception as e:
    print(f'Eager Failed! : {e}')
    import sys; sys.exit(1)

# Compiled Execution (tf.function)
# Analogous to torch.compile
print('Running Compiled (tf.function)...')
try:
    compiled_program = tf.function(fuzzed_program)
    out_compiled = compiled_program(sp_input_1, sp_input_2)
    dense_compiled = tf.sparse.to_dense(out_compiled)
    print('Compile Success! ')
except Exception as e:
    print(f'Compile Failed! : {e}')
    import sys; sys.exit(1)

# Verification
# Check if shapes match
if dense_eager.shape != dense_compiled.shape:
    print(f' Shape mismatch: Eager {dense_eager.shape} vs Compiled {dense_compiled.shape}')
    import sys; sys.exit(1)

# Check if values match
# Using numpy for comparison
diff = np.abs(dense_eager.numpy() - dense_compiled.numpy())
if np.any(diff > 0):
    print(f' Forward outputs differ!')
    print('Eager output:', dense_eager.numpy())
    print('Compiled output:', dense_compiled.numpy())
    print('Absolute diff:', diff)
    import sys; sys.exit(1)
else:
    print(' Outputs match between Eager and Compiled modes.')