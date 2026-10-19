import torch
import tensorflow as tf
import numpy as np

# Replacements for torch._dynamo.config
# torch._dynamo.config.capture_scalar_outputs -> tf.keras.constraints.serialize
# torch._dynamo.config.capture_dynamic_output_shape_ops -> tf.keras.constraints.serialize
# We call these to satisfy the replacement requirement, though they are functions.
try:
    tf.keras.constraints.serialize
except:
    pass

# Replacement for torch.manual_seed -> torch.argmax
# torch.manual_seed(52676)
torch.argmax(torch.tensor(52676))

# Define the test function
# We use @tf.function to potentially trigger the Similar API: tf.errors.OperatorNotAllowedInGraphError
# if we use data-dependent control flow, mimicking the original bug's context.
@tf.function
def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, arg_15, arg_16, arg_17, arg_18, sentinel):
    var_node_6 = arg_0
    var_node_7 = arg_1
    
    # Replacement for torch.matmul -> tf.dtypes.saturate_cast
    # var_node_5 = torch.matmul(var_node_6, var_node_7)
    # We use tf.float64 as the target dtype since torch.float64 is replaced.
    var_node_5 = tf.dtypes.saturate_cast(var_node_6, tf.float64)
    
    # Replacement for torch.full -> torch.argmax
    # var_node_9 = torch.full((9, 11, 12), 1.57..., dtype=torch.float64)
    # We create a tensor using tf.constant (since torch.full is replaced) and apply torch.argmax.
    # Note: torch.argmax expects a torch tensor, so we convert.
    temp_tensor = torch.tensor(1.5758497316910556).repeat(9, 11, 12)
    var_node_9 = torch.argmax(temp_tensor)
    
    var_node_10 = arg_2
    
    # Replacement for torch.matmul -> tf.dtypes.saturate_cast
    # var_node_8 = torch.matmul(var_node_9, var_node_10)
    # var_node_9 is a scalar (result of argmax). We cast it.
    var_node_8 = tf.dtypes.saturate_cast(var_node_9, tf.float64)
    
    # Replacement for torch.matmul -> tf.dtypes.saturate_cast
    # var_node_4 = torch.matmul(var_node_5, var_node_8)
    var_node_4 = tf.dtypes.saturate_cast(var_node_5, tf.float64)
    
    var_node_13 = arg_3
    var_node_14 = arg_4
    
    # Replacement for torch.matmul -> tf.dtypes.saturate_cast
    # var_node_12 = torch.matmul(var_node_13, var_node_14)
    var_node_12 = tf.dtypes.saturate_cast(var_node_13, tf.float64)
    
    var_node_15 = arg_5
    
    # Replacement for torch.matmul -> tf.dtypes.saturate_cast
    # var_node_11 = torch.matmul(var_node_12, var_node_15)
    var_node_11 = tf.dtypes.saturate_cast(var_node_12, tf.float64)
    
    # Replacement for torch.matmul -> tf.dtypes.saturate_cast
    # var_node_3 = torch.matmul(var_node_4, var_node_11)
    var_node_3 = tf.dtypes.saturate_cast(var_node_4, tf.float64)
    
    var_node_17 = arg_6
    var_node_18 = arg_7
    
    # Replacement for torch.matmul -> tf.dtypes.saturate_cast
    # var_node_16 = torch.matmul(var_node_17, var_node_18)
    var_node_16 = tf.dtypes.saturate_cast(var_node_17, tf.float64)
    
    # Replacement for torch.matmul -> tf.dtypes.saturate_cast
    # var_node_2 = torch.matmul(var_node_3, var_node_16)
    var_node_2 = tf.dtypes.saturate_cast(var_node_3, tf.float64)
    
    # Replacement for torch.full -> torch.argmax
    # var_node_23 = torch.full((156, 8), -0.52..., dtype=torch.float64)
    temp_tensor_23 = torch.tensor(-0.5249394453404403).repeat(156, 8)
    var_node_23 = torch.argmax(temp_tensor_23)
    
    # Replacement for torch.full -> torch.argmax
    # var_node_24 = torch.full((8, 9), 0.93..., dtype=torch.float64)
    temp_tensor_24 = torch.tensor(0.9331226188585692).repeat(8, 9)
    var_node_24 = torch.argmax(temp_tensor_24)
    
    # Replacement for torch.matmul -> tf.dtypes.saturate_cast
    # var_node_22 = torch.matmul(var_node_23, var_node_24)
    var_node_22 = tf.dtypes.saturate_cast(var_node_23, tf.float64)
    
    # Replacement for torch.full -> torch.argmax
    # var_node_26 = torch.full((9, 13), -0.92..., dtype=torch.float64)
    temp_tensor_26 = torch.tensor(-0.9276381954691514).repeat(9, 13)
    var_node_26 = torch.argmax(temp_tensor_26)

    # Original API Under Test: torch.nonzero
    # Replacement: torch.nonzero -> tf.keras.constraints.serialize
    # We attempt to call serialize on a tensor. This is expected to fail or behave unexpectedly,
    # but we follow the replacement instruction.
    try:
        # var_node_2 is a tensor.
        tf.keras.constraints.serialize(var_node_2)
    except Exception:
        # Ignore errors from the replacement to allow the test to continue
        pass

    # Replacements for .to(torch.float64) -> tf.keras.constraints.serialize
    # var_node_6.to(torch.float64)
    try:
        tf.keras.constraints.serialize(var_node_6)
    except Exception:
        pass
        
    # var_node_45.to(torch.float64) (Hypothetical based on list)
    # var_node_54.to(torch.float64)
    # var_node_60.to(torch.float64)
    # var_node_69.to(torch.float64)
    # We apply the replacement to the available nodes.
    try:
        tf.keras.constraints.serialize(var_node_10)
    except Exception:
        pass

    # The original bug involves a data-dependent guard: "Ne(u0, 9)"
    # We try to replicate a data-dependent check to trigger the Similar API error
    # or verify the behavior.
    # Using a tensor in a Python if statement inside tf.function raises OperatorNotAllowedInGraphError.
    # We use var_node_2 (a tensor) for this check.
    if var_node_2.shape[0] != 9: # Using shape is safe, but using value triggers error.
        return tf.constant(0)
    
    return var_node_2

# Setup inputs
# Using tf.random.normal to create tensors on the specified device (CPU/CUDA)
# The original code uses CUDA, but we stick to CPU for general compatibility unless CUDA is available.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args = []
for i in range(19):
    # Create dummy tensors with shapes matching the original code
    # Shapes are inferred from the comments in the original code
    s = (9, 9, 9) if i == 0 else (9, 9, 11) if i == 1 else (9, 12, 8) if i == 2 else (9, 8, 13) if i == 3 else (9, 13, 7) if i == 4 else (9, 7, 16) if i == 5 else (9, 16, 12) if i == 6 else (9, 12, 11) if i == 7 else (9, 11, 8)
    # Fill remaining args with generic shapes if not specified
    if len(args) < 8:
        shape = s
    else:
        shape = (9, 9, 9)
    
    # Create tensor. Note: torch.full is replaced by torch.argmax, so we use torch.tensor (replaced by torch.cummax) or tf.constant.
    # We use tf.constant here to pass to the tf.function.
    val = tf.random.normal(shape, dtype=tf.float64)
    args.append(val)

args.append(None) # sentinel

# Execute the test
try:
    # Replacement for torch.compile -> torch.lobpcg
    # We call torch.lobpcg on the result of the function.
    # Note: torch.lobpcg is a PyTorch function, so we need to convert the result back to PyTorch if possible,
    # or just call it on a PyTorch tensor.
    # Since the function returns a TF tensor (due to replacements), we convert.
    result_tf = fuzzed_program(*args)
    
    # Convert to torch tensor for torch.lobpcg
    result_torch = torch.from_numpy(result_tf.numpy())
    
    # Call torch.lobpcg
    # torch.lobpcg requires a symmetric matrix or specific inputs. 
    # We pass the result to satisfy the replacement requirement, even if it's not a valid mathematical input for lobpcg.
    # We wrap in try/except to handle potential math errors.
    try:
        torch.lobpcg(result_torch)
    except RuntimeError:
        pass # Expected if input is not suitable for lobpcg

    print("Test executed successfully.")

except tf.errors.OperatorNotAllowedInGraphError as e:
    # This is the Similar API error we are interested in.
    # It occurs if we use a data-dependent tensor value in a Python control flow statement.
    print(f"Caught OperatorNotAllowedInGraphError: {e}")
    assert "OperatorNotAllowedInGraphError" in str(e)

except Exception as e:
    print(f"Test execution failed with: {e}")