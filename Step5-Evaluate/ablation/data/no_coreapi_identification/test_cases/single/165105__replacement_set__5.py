import torch
import tensorflow as tf

# Configuration
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

# Replacement: torch.manual_seed -> torch.isinf
# Original: torch.manual_seed(70609)
torch.isinf(torch.tensor(70609))

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, arg_15, arg_16, arg_17, arg_18, arg_19, arg_20, arg_21, arg_22, arg_23, arg_24, arg_25, arg_26, sentinel):
    # Replacement: torch.full -> torch.amax
    # Original: var_node_4 = torch.full((14,), 1.2255859375, dtype=torch.float16)
    var_node_4 = torch.amax(torch.randn(14, dtype=torch.float16))
    
    var_node_6 = arg_0
    
    # Replacement: torch.full -> torch.amax
    # Original: var_node_9 = torch.full((6, 13), 1.3154296875, dtype=torch.float16)
    var_node_9 = torch.amax(torch.randn(6, 13, dtype=torch.float16))
    
    var_node_10 = arg_1
    
    # Original API Under Test: torch.matmul (Similar API: torch.matmul)
    var_node_8 = torch.matmul(var_node_9.to(torch.float16), var_node_10.to(torch.float16))
    
    var_node_12 = arg_2
    
    # Replacement: torch.full -> torch.amax
    # Original: var_node_13 = torch.full((10, 416), 0.1331787109375, dtype=torch.float16)
    var_node_13 = torch.amax(torch.randn(10, 416, dtype=torch.float16))
    
    var_node_11 = torch.matmul(var_node_12.to(torch.float16), var_node_13.to(torch.float16))
    var_node_7 = torch.matmul(var_node_8.to(torch.float16), var_node_11.to(torch.float16))
    var_node_5 = torch.matmul(var_node_6.to(torch.float16), var_node_7.to(torch.float16))
    var_node_3 = torch.matmul(var_node_4.to(torch.float16), var_node_5.to(torch.float16))
    var_node_2 = var_node_3.contiguous().view([26, 1, 16])
    
    var_node_16 = arg_3
    var_node_20 = arg_4
    var_node_19 = var_node_20.contiguous().view([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    
    # Replacement: torch.unsqueeze -> tf.experimental.dtensor.create_mesh
    # Original: var_node_18 = torch.unsqueeze(var_node_19, dim=15)
    var_node_18 = tf.experimental.dtensor.create_mesh(['x'], devices=['cpu:0'])
    
    # Replacement: torch.zeros -> torch.norm
    # Original: _x_nz = torch.zeros((26, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), dtype=torch.bool, device=var_node_18.device)
    # Note: var_node_18 is now a Mesh, not a Tensor, so .device access might fail or behave differently.
    # We use a dummy tensor for norm input.
    _x_nz = torch.norm(torch.randn(26, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))
    
    _x_nz_flat = _x_nz.reshape(-1)
    _x_nz_flat[:26] = True
    
    # Replacement: torch.nonzero -> torch.take
    # Original: var_node_17 = torch.nonzero(_x_nz)
    var_node_17 = torch.take(_x_nz, torch.tensor([0]))
    
    # Replacement: torch.clamp -> torch.kron
    # Replacement: torch.nn.functional.embedding -> torch.nn.functional.binary_cross_entropy
    # Original: var_node_15 = torch.nn.functional.embedding(torch.clamp(var_node_17.to(torch.int64), 0, var_node_16.size(0)-1), var_node_16)
    clamped_indices = torch.kron(var_node_17.to(torch.float32), torch.tensor([1.0]))
    var_node_15 = torch.nn.functional.binary_cross_entropy(clamped_indices, var_node_16)
    
    return var_node_15

if __name__ == "__main__":
    # Create dummy arguments for the function to be runnable
    # Using CPU to ensure runnability without CUDA
    args = [
        torch.randn(14, 6, dtype=torch.float16), # arg_0
        torch.randn(13, 1, dtype=torch.float16), # arg_1
        torch.randn(1, 10, dtype=torch.float16), # arg_2
        torch.randn(329, 4, dtype=torch.float16), # arg_3
        torch.tensor([True]), # arg_4
        torch.randn(26, 4, 8, dtype=torch.float16), # arg_5
    ] + [torch.randn(1, dtype=torch.float16) for _ in range(21)] # arg_6 to arg_26
    
    try:
        result = fuzzed_program(*args, sentinel=None)
        print("Test case executed. Result:", result)
    except Exception as e:
        print(f"Test case executed with expected errors due to API replacements: {e}")