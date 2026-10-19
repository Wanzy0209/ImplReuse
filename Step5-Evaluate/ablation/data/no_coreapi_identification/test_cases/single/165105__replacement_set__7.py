import torch
import tensorflow as tf

# Setup configurations
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

# Replacement: torch.manual_seed -> torch.argmax
# Original: torch.manual_seed(70609)
torch.argmax(torch.tensor([70609]))

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, arg_15, arg_16, arg_17, arg_18, arg_19, arg_20, arg_21, arg_22, arg_23, arg_24, arg_25, arg_26, sentinel):
    # Replacement: torch.full -> torch.argmax
    # Original: var_node_4 = torch.full((14,), 1.2255859375, dtype=torch.float16)
    var_node_4 = torch.argmax(torch.randn(14), dim=0).to(torch.float16)

    var_node_6 = arg_0

    # Replacement: torch.full -> torch.argmax
    # Original: var_node_9 = torch.full((6, 13), 1.3154296875, dtype=torch.float16)
    var_node_9 = torch.argmax(torch.randn(6, 13), dim=0).to(torch.float16)

    var_node_10 = arg_1
    var_node_8 = torch.matmul(var_node_9.to(torch.float16), var_node_10.to(torch.float16))

    var_node_12 = arg_2

    # Replacement: torch.full -> torch.argmax
    # Original: var_node_13 = torch.full((10, 416), 0.1331787109375, dtype=torch.float16)
    var_node_13 = torch.argmax(torch.randn(10, 416), dim=1).to(torch.float16)

    var_node_11 = torch.matmul(var_node_12.to(torch.float16), var_node_13.to(torch.float16))
    var_node_7 = torch.matmul(var_node_8.to(torch.float16), var_node_11.to(torch.float16))
    var_node_5 = torch.matmul(var_node_6.to(torch.float16), var_node_7.to(torch.float16))
    var_node_3 = torch.matmul(var_node_4.to(torch.float16), var_node_5.to(torch.float16))
    var_node_2 = var_node_3.contiguous().view([26, 1, 16])

    var_node_16 = arg_3
    var_node_20 = arg_4
    var_node_19 = var_node_20.contiguous().view([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    # Replacement: torch.unsqueeze -> torch.index_select
    # Original: var_node_18 = torch.unsqueeze(var_node_19, dim=15)
    var_node_18 = torch.index_select(var_node_19, dim=0, index=torch.tensor([0]))

    # Replacement: torch.zeros -> torch.slogdet
    # Original: _x_nz = torch.zeros((26, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), dtype=torch.bool, device=var_node_18.device)
    # slogdet requires a square matrix. We create a dummy one and take the sign tensor to maintain flow.
    _dummy_matrix = torch.randn(26, 26, dtype=torch.float16, device=var_node_18.device)
    _x_nz = torch.slogdet(_dummy_matrix)[0]

    _x_nz_flat = _x_nz.reshape(-1)
    _x_nz_flat[:26] = True

    # Replacement: torch.nonzero -> torch.clamp_min
    # Original: var_node_17 = torch.nonzero(_x_nz)
    var_node_17 = torch.clamp_min(_x_nz, 0)

    # Replacement: torch.clamp -> torch.tril_indices
    # Original: indices = torch.clamp(var_node_17.to(torch.int64), 0, var_node_16.size(0)-1)
    indices = torch.tril_indices(var_node_16.size(0), var_node_16.size(0), offset=0, device=var_node_17.device)

    # Replacement: torch.nn.functional.embedding -> torch.nn.utils.clip_grad_norm_
    # Original: var_node_15 = torch.nn.functional.embedding(indices, var_node_16)
    var_node_15 = torch.nn.utils.clip_grad_norm_(var_node_16, 1.0)

    var_node_25 = arg_5
    
    return var_node_15

if __name__ == "__main__":
    # Setup dummy arguments to make the test case runnable
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    arg_0 = torch.randn(14, 6, dtype=torch.float16, device=device)
    arg_1 = torch.randn(13, 1, dtype=torch.float16, device=device)
    arg_2 = torch.randn(1, 10, dtype=torch.float16, device=device)
    arg_3 = torch.randn(329, 4, dtype=torch.float16, device=device)
    arg_4 = torch.tensor([True], dtype=torch.bool, device=device)
    arg_5 = torch.randn(26, 4, 8, dtype=torch.float16, device=device)
    
    # Fill remaining args with dummy tensors
    args = [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5] + [torch.randn(1, device=device) for _ in range(21)]
    
    try:
        result = fuzzed_program(*args, sentinel=None)
        print("Test case executed successfully.")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Test case execution failed (expected due to API replacement semantics): {e}")