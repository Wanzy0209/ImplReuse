import torch
import tensorflow as tf

# Original configuration
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

# Replacement: torch.manual_seed -> tf.compat.v1.assign_add
# Original: torch.manual_seed(70609)
tf_var = tf.Variable(0.0)
tf.compat.v1.assign_add(tf_var, 70609)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, arg_15, arg_16, arg_17, arg_18, arg_19, arg_20, arg_21, arg_22, arg_23, arg_24, arg_25, arg_26, sentinel):
    # Replacement: torch.full -> torch.amin
    # Original: var_node_4 = torch.full((14,), 1.2255859375, dtype=torch.float16)
    # To keep shape (14,), we take amin over a dummy dimension
    var_node_4 = torch.amin(torch.randn(14, 1, dtype=torch.float16), dim=1) # size=(14,), stride=(1,), dtype=float16, device=cuda
    
    var_node_6 = arg_0 # size=(14, 6), stride=(6, 1), dtype=float16, device=cuda
    
    # Replacement: torch.full -> torch.amin
    # Original: var_node_9 = torch.full((6, 13), 1.3154296875, dtype=torch.float16)
    var_node_9 = torch.amin(torch.randn(6, 13, 1, dtype=torch.float16), dim=2) # size=(6, 13), stride=(13, 1), dtype=float16, device=cuda
    
    var_node_10 = arg_1 # size=(13, 1), stride=(1, 1), dtype=float16, device=cuda
    var_node_8 = torch.matmul(var_node_9.to(torch.float16), var_node_10.to(torch.float16)) # size=(6, 1), stride=(1, 1), dtype=float16, device=cuda
    
    var_node_12 = arg_2 # size=(1, 10), stride=(10, 1), dtype=float16, device=cuda
    
    # Replacement: torch.full -> torch.amin
    # Original: var_node_13 = torch.full((10, 416), 0.1331787109375, dtype=torch.float16)
    var_node_13 = torch.amin(torch.randn(10, 416, 1, dtype=torch.float16), dim=2) # size=(10, 416), stride=(416, 1), dtype=float16, device=cuda
    
    var_node_11 = torch.matmul(var_node_12.to(torch.float16), var_node_13.to(torch.float16)) # size=(1, 416), stride=(416, 1), dtype=float16, device=cuda
    var_node_7 = torch.matmul(var_node_8.to(torch.float16), var_node_11.to(torch.float16)) # size=(6, 416), stride=(416, 1), dtype=float16, device=cuda
    var_node_5 = torch.matmul(var_node_6.to(torch.float16), var_node_7.to(torch.float16)) # size=(14, 416), stride=(416, 1), dtype=float16, device=cuda
    var_node_3 = torch.matmul(var_node_4.to(torch.float16), var_node_5.to(torch.float16)) # size=(416,), stride=(1,), dtype=float16, device=cuda
    var_node_2 = var_node_3.contiguous().view([26, 1, 16]) # size=(26, 1, 16), stride=(16, 16, 1), dtype=float16, device=cuda
    
    var_node_16 = arg_3 # size=(329, 4), stride=(4, 1), dtype=float16, device=cuda
    var_node_20 = arg_4 # size=(1,), stride=(1,), dtype=bool, device=cuda
    var_node_19 = var_node_20.contiguous().view([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) # size=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), stride=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), dtype=bool, device=cuda
    
    # Replacement: torch.unsqueeze -> torch.diff
    # Original: var_node_18 = torch.unsqueeze(var_node_19, dim=15)
    var_node_18 = torch.diff(var_node_19) # size changes based on diff behavior
    
    # Replacement: torch.zeros -> torch.normal
    # Original: _x_nz = torch.zeros((26, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), dtype=torch.bool, device=var_node_18.device)
    _x_nz = torch.normal(0, 1, size=(26, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), device=var_node_18.device)
    
    _x_nz_flat = _x_nz.reshape(-1)
    _x_nz_flat[:26] = 1.0 # Assigning float to float tensor
    
    # Replacement: torch.nonzero -> torch.mv
    # Original: var_node_17 = torch.nonzero(_x_nz)
    # _x_nz is float now. mv requires matrix and vector.
    # Reshape _x_nz to (26, 1) to act as matrix, create dummy vector of size 1
    var_node_17 = torch.mv(_x_nz.reshape(26, 1), torch.randn(1, device=_x_nz.device)) # size=(26,)
    
    # Replacement: torch.nn.functional.embedding -> torch.nn.functional.kl_div
    # Replacement: torch.clamp -> torch.outer (inside the embedding call)
    # Original: torch.nn.functional.embedding(torch.clamp(var_node_17.to(torch.int64), 0, var_node_16.size(0)-1), var_node_16)
    # New: torch.nn.functional.kl_div(torch.outer(...), var_node_16)
    # Note: var_node_17 is float from mv. outer(var_node_17, dummy) -> (26, 1)
    # var_node_16 is (329, 4). Shapes mismatch for kl_div, but we perform the replacement.
    var_node_15 = torch.nn.functional.kl_div(
        torch.outer(var_node_17, torch.randn(1, device=var_node_17.device)), 
        var_node_16
    ) # size=(26, 16, 4), stride=(64, 4, 1), dtype=float16, device=cuda (Original comment, actual shape will differ)
    
    var_node_25 = arg_5 # size=(26, 4, 8),
    
    return var_node_15

# Generate dummy arguments to make the test case runnable
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    arg_0 = torch.randn(14, 6, dtype=torch.float16, device=device)
    arg_1 = torch.randn(13, 1, dtype=torch.float16, device=device)
    arg_2 = torch.randn(1, 10, dtype=torch.float16, device=device)
    arg_3 = torch.randn(329, 4, dtype=torch.float16, device=device)
    arg_4 = torch.tensor([True], device=device)
    arg_5 = torch.randn(26, 4, 8, dtype=torch.float16, device=device)
    
    # Fill remaining args with None or dummy tensors
    args = [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5] + [None] * 21
    
    try:
        output = fuzzed_program(*args, sentinel=None)
        print("Test executed. Output shape:", output.shape if hasattr(output, 'shape') else output)
    except Exception as e:
        print(f"Test executed with exception (expected due to semantic replacements): {e}")