import torch
import tensorflow as tf
import torch.nn.functional as F

# Replacement for torch.manual_seed(13653)
# tf.compat.v1.assign_add requires a variable to operate on.
# We create a dummy variable and perform the operation to satisfy the replacement.
tf_var = tf.Variable(0.0)
tf.compat.v1.assign_add(tf_var, 13653)

def fuzzed_program(arg_0, arg_1, sentinel):
    # Replacement for torch.full((), 1.0, dtype=torch.float32, device=cuda)
    # torch.amin computes the minimum value. We apply it to the sentinel (which is 1.0)
    # to effectively retrieve the scalar value 1.0, preserving the logic flow.
    var_node_3 = torch.amin(sentinel)

    var_node_2 = var_node_3.item()
    var_node_5 = -3
    var_node_6 = arg_0
    var_node_4 = var_node_5 + var_node_6
    var_node_1 = var_node_2 + var_node_4
    var_node_9 = 1
    var_node_10 = -10
    var_node_8 = var_node_9 / var_node_10
    var_node_12 = arg_1
    var_node_13 = -5
    var_node_11 = var_node_12 / var_node_13
    var_node_7 = var_node_8 + var_node_11
    var_node_0 = var_node_1 * var_node_7
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
# Replacement for torch.tensor(1.0, requires_grad=True)
# tf.experimental.numpy.diag creates a diagonal matrix. We create a 1x1 matrix with 1.0,
# convert to torch, and extract the scalar.
tf_diag_sentinel = tf.experimental.numpy.diag([1.0])
sentinel = torch.from_numpy(tf_diag_sentinel.numpy())[0, 0].detach().clone()
sentinel.requires_grad = True

# Helper to generate arguments using the replacement APIs
def generate_arg():
    # Replacement for torch.randn(())
    # torch.nn.functional.adaptive_max_pool2d requires a 4D input.
    # We generate this input using the replacement for torch.tensor.
    
    # Replacement for torch.tensor(...)
    # Create a diagonal matrix using TensorFlow, convert to PyTorch
    t = tf.experimental.numpy.diag([2.0, 3.0]) 
    pt_t = torch.from_numpy(t.numpy())
    
    # Reshape to 4D (Batch, Channel, Height, Width) for pooling
    pt_t = pt_t.unsqueeze(0).unsqueeze(0)
    
    # Apply adaptive_max_pool2d to reduce to a 1x1 output
    pooled = F.adaptive_max_pool2d(pt_t, (1, 1))
    
    # Extract item and cast to int (mimicking original dtype=torch.int64)
    return int(pooled.item())

arg_0 = generate_arg()
arg_1 = generate_arg()

args = (arg_0, arg_1) + (sentinel,)
result_original = fuzzed_program(*args)
print(' eager success')
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print(' compile success')