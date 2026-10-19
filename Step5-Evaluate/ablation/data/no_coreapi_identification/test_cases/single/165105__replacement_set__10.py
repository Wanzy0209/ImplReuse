import torch
import torch.nn.functional as F
import tensorflow as tf
from torch.distributions import constraints

# Configuration from original test case
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

# Replacement: torch.manual_seed -> tf.compat.v1.assert_variables_initialized
# Note: This is a cross-library replacement.
try:
    tf.compat.v1.assert_variables_initialized()
except Exception as e:
    # In TF2 eager mode, this might behave differently or raise if no session.
    # We catch it to ensure the PyTorch part of the test can proceed if TF setup is minimal.
    pass

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, arg_15, arg_16, arg_17, arg_18, arg_19, arg_20, arg_21, arg_22, arg_23, arg_24, arg_25, arg_26, sentinel):
    # Replacement: torch.full -> torch.rand_like
    # Original: var_node_4 = torch.full((14,), 1.2255859375, dtype=torch.float16)
    var_node_4 = torch.rand_like(torch.empty(14, dtype=torch.float16))
    
    var_node_6 = arg_0
    
    # Replacement: torch.full -> torch.rand_like
    # Original: var_node_9 = torch.full((6, 13), 1.3154296875, dtype=torch.float16)
    var_node_9 = torch.rand_like(torch.empty(6, 13, dtype=torch.float16))
    
    var_node_10 = arg_1
    var_node_8 = torch.matmul(var_node_9.to(torch.float16), var_node_10.to(torch.float16))
    
    var_node_12 = arg_2
    
    # Replacement: torch.full -> torch.rand_like
    # Original: var_node_13 = torch.full((10, 416), 0.1331787109375, dtype=torch.float16)
    var_node_13 = torch.rand_like(torch.empty(10, 416, dtype=torch.float16))
    
    var_node_11 = torch.matmul(var_node_12.to(torch.float16), var_node_13.to(torch.float16))
    var_node_7 = torch.matmul(var_node_8.to(torch.float16), var_node_11.to(torch.float16))
    var_node_5 = torch.matmul(var_node_6.to(torch.float16), var_node_7.to(torch.float16))
    var_node_3 = torch.matmul(var_node_4.to(torch.float16), var_node_5.to(torch.float16))
    var_node_2 = var_node_3.contiguous().view([26, 1, 16])
    
    var_node_16 = arg_3
    var_node_20 = arg_4
    var_node_19 = var_node_20.contiguous().view([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    
    # Replacement: torch.unsqueeze -> torch.distributions.constraints.stack
    # Original: var_node_18 = torch.unsqueeze(var_node_19, dim=15)
    # Note: This changes the type from Tensor to Constraint object. 
    # Subsequent .device access might fail depending on implementation details of the constraint object.
    # We proceed with the replacement as requested.
    var_node_18 = constraints.stack([var_node_19], dim=15)
    
    # Replacement: torch.zeros -> torch.qr
    # Original: _x_nz = torch.zeros((26, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), dtype=torch.bool, device=var_node_18.device)
    # We create a random tensor of the same shape to pass to qr, and take the Q component.
    # We try to use the device from var_node_18 if it has one, otherwise default.
    device = var_node_18.device if hasattr(var_node_18, 'device') else torch.device('cuda')
    _x_nz_input = torch.randn((26, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), dtype=torch.float32, device=device)
    _x_nz = torch.qr(_x_nz_input)[0]
    
    _x_nz_flat = _x_nz.reshape(-1)
    _x_nz_flat[:26] = True
    
    # Replacement: torch.nonzero -> torch.argmin
    # Original: var_node_17 = torch.nonzero(_x_nz)
    var_node_17 = torch.argmin(_x_nz)
    
    # Replacement: torch.clamp -> torch.diag
    # Original: torch.clamp(var_node_17.to(torch.int64), 0, var_node_16.size(0)-1)
    clamped_indices = torch.diag(var_node_17.to(torch.int64))
    
    # Replacement: torch.nn.functional.embedding -> torch.nn.functional.feature_alpha_dropout
    # Original: torch.nn.functional.embedding(clamped_indices, var_node_16)
    # feature_alpha_dropout takes input and p. We pass the result of diag.
    var_node_15 = F.feature_alpha_dropout(clamped_indices, p=0.5)
    
    return var_node_15

# Setup dummy arguments to make the test case runnable
# Shapes inferred from the original test case comments
arg_0 = torch.randn(14, 6, dtype=torch.float16, device='cuda')
arg_1 = torch.randn(13, 1, dtype=torch.float16, device='cuda')
arg_2 = torch.randn(1, 10, dtype=torch.float16, device='cuda')
arg_3 = torch.randn(329, 4, dtype=torch.float16, device='cuda')
arg_4 = torch.tensor([True], dtype=torch.bool, device='cuda')
arg_5 = torch.randn(26, 4, 8, dtype=torch.float16, device='cuda') # Shape guessed based on context
# Fill remaining args with dummy tensors
args = [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5] + [torch.randn(1, device='cuda') for _ in range(21)]

# Run the test
try:
    output = fuzzed_program(*args, sentinel=None)
    print("Test case executed successfully.")
    print(f"Output shape: {output.shape}")
except Exception as e:
    print(f"Test case execution failed with error: {e}")