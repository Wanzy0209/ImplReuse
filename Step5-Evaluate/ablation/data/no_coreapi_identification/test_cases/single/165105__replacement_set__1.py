import torch
import torch.nn.functional as F

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# torch.manual_seed -> torch.set_rng_state
# We initialize a generator with the seed to get a state, then set that state.
gen = torch.manual_seed(70609)
torch.set_rng_state(gen.get_state())

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, arg_15, arg_16, arg_17, arg_18, arg_19, arg_20, arg_21, arg_22, arg_23, arg_24, arg_25, arg_26, sentinel):
    # torch.full -> torch.fmax
    # var_node_4 = torch.full((14,), 1.2255859375, dtype=torch.float16)
    var_node_4 = torch.fmax(torch.zeros((14,), dtype=torch.float16, device=device), torch.tensor(1.2255859375, dtype=torch.float16, device=device))
    
    var_node_6 = arg_0
    
    # torch.full -> torch.fmax
    # var_node_9 = torch.full((6, 13), 1.3154296875, dtype=torch.float16)
    var_node_9 = torch.fmax(torch.zeros((6, 13), dtype=torch.float16, device=device), torch.tensor(1.3154296875, dtype=torch.float16, device=device))
    
    var_node_10 = arg_1
    
    # Original API: torch.matmul (Similar API is also torch.matmul)
    var_node_8 = torch.matmul(var_node_9.to(torch.float16), var_node_10.to(torch.float16))
    
    var_node_12 = arg_2
    
    # torch.full -> torch.fmax
    # var_node_13 = torch.full((10, 416), 0.1331787109375, dtype=torch.float16)
    var_node_13 = torch.fmax(torch.zeros((10, 416), dtype=torch.float16, device=device), torch.tensor(0.1331787109375, dtype=torch.float16, device=device))
    
    var_node_11 = torch.matmul(var_node_12.to(torch.float16), var_node_13.to(torch.float16))
    var_node_7 = torch.matmul(var_node_8.to(torch.float16), var_node_11.to(torch.float16))
    var_node_5 = torch.matmul(var_node_6.to(torch.float16), var_node_7.to(torch.float16))
    var_node_3 = torch.matmul(var_node_4.to(torch.float16), var_node_5.to(torch.float16))
    var_node_2 = var_node_3.contiguous().view([26, 1, 16])
    
    var_node_16 = arg_3
    var_node_20 = arg_4
    var_node_19 = var_node_20.contiguous().view([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    
    # torch.unsqueeze -> torch.cholesky_solve
    # var_node_18 = torch.unsqueeze(var_node_19, dim=15)
    # To mimic unsqueeze adding a dimension, we use cholesky_solve which returns (..., N).
    # We construct inputs such that it effectively acts as an identity operation on the shape.
    # var_node_19 is bool, cast to float for cholesky_solve.
    # input2 must be (..., N, N). We use Identity matrix.
    batch_shape = var_node_19.shape
    eye = torch.eye(1, device=device, dtype=torch.float32)
    input2 = eye.view(*([1]*len(batch_shape)), 1, 1).expand(*batch_shape, 1, 1)
    var_node_18 = torch.cholesky_solve(var_node_19.float().unsqueeze(-1), input2)
    
    # torch.zeros -> torch.ones
    # _x_nz = torch.zeros((26, 1, 1, ...), dtype=torch.bool, device=...)
    _x_nz = torch.ones((26, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), dtype=torch.bool, device=device)
    
    _x_nz_flat = _x_nz.reshape(-1)
    _x_nz_flat[:26] = True
    
    # torch.nonzero -> torch.nn.functional.hardshrink
    # var_node_17 = torch.nonzero(_x_nz)
    # hardshrink requires float input.
    var_node_17 = F.hardshrink(_x_nz.float())
    
    # torch.nn.functional.embedding -> torch.nn.functional.embedding_bag
    # torch.clamp -> torch.clip
    # var_node_15 = torch.nn.functional.embedding(torch.clamp(var_node_17.to(torch.int64), 0, var_node_16.size(0)-1), var_node_16)
    indices = torch.clip(var_node_17.long(), 0, var_node_16.size(0)-1)
    var_node_15 = F.embedding_bag(indices, var_node_16, mode='sum')
    
    # The rest of the program is omitted as it was cut off in the issue description.
    return var_node_3, var_node_15

# Create dummy arguments to make the test case runnable
# Shapes and dtypes are inferred from the comments in the bug report
args = [
    torch.randn(14, 6, dtype=torch.float16, device=device), # arg_0
    torch.randn(13, 1, dtype=torch.float16, device=device), # arg_1
    torch.randn(1, 10, dtype=torch.float16, device=device), # arg_2
    torch.randn(329, 4, dtype=torch.float16, device=device), # arg_3
    torch.tensor([True], device=device),                    # arg_4
    torch.randn(26, 4, 8, dtype=torch.float16, device=device) # arg_5
]

# Fill remaining args with dummy tensors
for _ in range(21):
    args.append(torch.randn(1, device=device))

# Run the test
try:
    result = fuzzed_program(*args, None)
    print("Test case executed successfully.")
    print(f"Result var_node_3 shape: {result[0].shape}")
    print(f"Result var_node_15 shape: {result[1].shape}")
except Exception as e:
    print(f"Test case failed with error: {e}")