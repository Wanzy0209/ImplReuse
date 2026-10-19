import torch
import tensorflow as tf

# Original configuration
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

# Replacement: torch.manual_seed -> tf.compat.v1.assign_sub
# Original: torch.manual_seed(70609)
# Replacement:
tf_var = tf.Variable(70609)
tf.compat.v1.assign_sub(tf_var, 0)

# Setup device and dummy arguments to make the test runnable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dummy_args():
    # Create a list of 27 arguments + 1 sentinel
    # Based on the shapes found in the original test case
    args = [
        torch.randn(14, 6, dtype=torch.float16, device=device),  # arg_0
        torch.randn(13, 1, dtype=torch.float16, device=device),  # arg_1
        torch.randn(1, 10, dtype=torch.float16, device=device),  # arg_2
        torch.randn(329, 4, dtype=torch.float16, device=device), # arg_3
        torch.tensor([True], dtype=torch.bool, device=device),   # arg_4
        torch.randn(26, 4, 8, dtype=torch.float16, device=device) # arg_5
    ]
    # Fill the rest with generic tensors
    for _ in range(21): # arg_6 to arg_26
        args.append(torch.randn(10, 10, dtype=torch.float16, device=device))
    args.append(None) # sentinel
    return args

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, arg_15, arg_16, arg_17, arg_18, arg_19, arg_20, arg_21, arg_22, arg_23, arg_24, arg_25, arg_26, sentinel):
    # Replacement: torch.full -> torch.randint
    # Original: var_node_4 = torch.full((14,), 1.2255859375, dtype=torch.float16)
    var_node_4 = torch.randint(0, 10, (14,), dtype=torch.float16, device=device)
    
    var_node_6 = arg_0
    
    # Replacement: torch.full -> torch.randint
    # Original: var_node_9 = torch.full((6, 13), 1.3154296875, dtype=torch.float16)
    var_node_9 = torch.randint(0, 10, (6, 13), dtype=torch.float16, device=device)
    
    var_node_10 = arg_1
    var_node_8 = torch.matmul(var_node_9.to(torch.float16), var_node_10.to(torch.float16))
    
    var_node_12 = arg_2
    
    # Replacement: torch.full -> torch.randint
    # Original: var_node_13 = torch.full((10, 416), 0.1331787109375, dtype=torch.float16)
    var_node_13 = torch.randint(0, 10, (10, 416), dtype=torch.float16, device=device)
    
    var_node_11 = torch.matmul(var_node_12.to(torch.float16), var_node_13.to(torch.float16))
    var_node_7 = torch.matmul(var_node_8.to(torch.float16), var_node_11.to(torch.float16))
    var_node_5 = torch.matmul(var_node_6.to(torch.float16), var_node_7.to(torch.float16))
    var_node_3 = torch.matmul(var_node_4.to(torch.float16), var_node_5.to(torch.float16))
    var_node_2 = var_node_3.contiguous().view([26, 1, 16])
    
    var_node_16 = arg_3
    var_node_20 = arg_4
    var_node_19 = var_node_20.contiguous().view([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    
    # Replacement: torch.unsqueeze -> torch.stack
    # Original: var_node_18 = torch.unsqueeze(var_node_19, dim=15)
    var_node_18 = torch.stack([var_node_19], dim=15)
    
    # Replacement: torch.zeros -> torch.linspace
    # Original: _x_nz = torch.zeros((26, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), dtype=torch.bool, device=var_node_18.device)
    _x_nz = torch.linspace(0, 1, steps=26).view(26, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1).to(torch.bool).to(var_node_18.device)
    
    _x_nz_flat = _x_nz.reshape(-1)
    _x_nz_flat[:26] = True
    
    # Replacement: torch.nonzero -> torch.floor_divide
    # Original: var_node_17 = torch.nonzero(_x_nz)
    var_node_17 = torch.floor_divide(_x_nz, 1)
    
    # Replacement: torch.clamp -> torch.vander
    # Replacement: torch.nn.functional.embedding -> torch.nn.functional.binary_cross_entropy_with_logits
    # Original: var_node_15 = torch.nn.functional.embedding(torch.clamp(var_node_17.to(torch.int64), 0, var_node_16.size(0)-1), var_node_16)
    # Note: Applying replacements sequentially. 
    # 1. torch.clamp(...) becomes torch.vander(...)
    # 2. torch.nn.functional.embedding(...) becomes torch.nn.functional.binary_cross_entropy_with_logits(...)
    
    # Step 1 replacement: torch.clamp(var_node_17.to(torch.int64), 0, var_node_16.size(0)-1)
    # becomes torch.vander(var_node_17.to(torch.int64))
    # Note: vander usually expects float, but we follow the replacement instruction.
    vander_output = torch.vander(var_node_17.to(torch.float32))
    
    # Step 2 replacement: torch.nn.functional.embedding(input, weight)
    # becomes torch.nn.functional.binary_cross_entropy_with_logits(input, target)
    # We use the vander_output as both input and target to satisfy the signature.
    var_node_15 = torch.nn.functional.binary_cross_entropy_with_logits(vander_output, vander_output)
    
    var_node_25 = arg_5
    
    return var_node_15

# Execute the test
if __name__ == "__main__":
    args = get_dummy_args()
    try:
        result = fuzzed_program(*args)
        print("Test case executed successfully.")
        # Basic assertion to ensure output is a tensor
        assert isinstance(result, torch.Tensor)
    except Exception as e:
        print(f"Test case execution failed with error: {e}")