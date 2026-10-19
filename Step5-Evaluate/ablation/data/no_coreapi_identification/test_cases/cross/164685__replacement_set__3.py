import tensorflow as tf
import torch
import torch.nn.functional as F

# Similar API: tf.keras.initializers.RandomUniform
# Leveraging this to generate input data as per the "reuse" requirement.
initializer = tf.keras.initializers.RandomUniform(seed=19989)

# Replacement: torch.manual_seed -> tf.compat.v1.assign_sub
# We create a variable to simulate the seed state and perform the assignment.
seed_var = tf.Variable(19989, dtype=tf.int32)
tf.compat.v1.assign_sub(seed_var, 0)

# Replacement: torch._dynamo.config.capture_scalar_outputs -> tf.is_symbolic_tensor
# Replacement: torch._dynamo.config.capture_dynamic_output_shape_ops -> tf.is_symbolic_tensor
# We perform the check on a dummy tensor to satisfy the replacement requirement.
_ = tf.is_symbolic_tensor(tf.constant(1.0))

def fuzzed_program(arg_0, sentinel):
    var_node_2 = -6  # dtype=int64
    var_node_3 = arg_0  # dtype=int32
    var_node_1 = var_node_2 * var_node_3  # dtype=int32
    
    # Replacement: torch.full -> torch.randint
    # Original: var_node_5 = torch.full((), 1, dtype=torch.int64)
    var_node_5 = torch.randint(1, 2, ())
    
    # Replacement: var_node_5.item -> tf.is_symbolic_tensor
    # Original: var_node_4 = var_node_5.item()
    var_node_4 = tf.is_symbolic_tensor(var_node_5)
    
    var_node_0 = var_node_1 / var_node_4  # dtype=int64
    
    result = var_node_0 * sentinel
    
    # Replacement: result.is_complex -> tf.is_symbolic_tensor
    # Original: if result.is_complex():
    if tf.is_symbolic_tensor(result):
        # Original: result = result.real
        # No replacement specified for .real, keeping logic minimal
        pass
            
    return result

# Sentinel tensor
# Original: sentinel = torch.tensor(1.0, requires_grad=True)
# Replacement: torch.tensor -> tf.compat.v1.linalg.diag
sentinel = tf.compat.v1.linalg.diag([1.0])

# Argument generation
# Original: arg_0 = torch.tensor(torch.randn(()), dtype=torch.int32).item()

# Step 1: torch.randn(())
# Replacement: torch.randn -> torch.nn.functional.adaptive_max_pool1d
# We need an input for the pool. Use the Similar API (RandomUniform).
# Note: adaptive_max_pool1d expects 3D input (N, C, L).
tf_input = initializer(shape=(1, 1, 1))
torch_input = torch.from_numpy(tf_input.numpy())
pooled = torch.nn.functional.adaptive_max_pool1d(torch_input, output_size=1)

# Step 2: torch.tensor(...)
# Replacement: torch.tensor -> tf.compat.v1.linalg.diag
# pooled is a torch tensor. We need to convert to numpy/tf for diag.
diag_tensor = tf.compat.v1.linalg.diag(pooled.numpy().flatten())

# Step 3: .item()
# Replacement: .item() -> tf.is_symbolic_tensor
arg_0 = tf.is_symbolic_tensor(diag_tensor)

args = (arg_0, sentinel)

# Eager execution
print('Running eager...')
try:
    result_original = fuzzed_program(*args)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')

# Compiled execution
# Original: compiled_program = torch.compile(...)
# Replacement: torch.compile -> tf.is_symbolic_tensor
# We wrap in tf.function to create a symbolic context, 
# and the replacement logic implies checking symbolic status.

@tf.function
def run_compiled(arg_0, sentinel):
    # The replacement for torch.compile is tf.is_symbolic_tensor.
    # We call it here to represent the "compilation" step logic.
    _ = tf.is_symbolic_tensor(arg_0)
    return fuzzed_program(arg_0, sentinel)

print('Running compiled...')
try:
    result_compiled = run_compiled(*args)
    print(' compile success')
except Exception as e:
    print(f' compile failed: {e}')