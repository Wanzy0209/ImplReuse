import torch

# Configuration from the bug report
torch._dynamo.config.capture_scalar_outputs = True
torch.manual_seed(751735337)

# Ensure we use CUDA if available as per the bug report
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

# Recreating the tensor setup from the fuzzer report
# arg_0: size=(15, 108, 4), stride=(432, 1, 4), dtype=int16
arg_0 = torch.randn(15, 108, 4, dtype=torch.int16, device=device)

# Reproduce the non-contiguous tensor var_node_1
# This sequence of operations creates a tensor with specific strides
var_node_4 = arg_0
var_node_3 = torch.chunk(var_node_4, 4, dim=1)[0]
var_node_2 = torch.chunk(var_node_3, 4, dim=2)[0]
var_node_1 = torch.squeeze(var_node_2)
# var_node_1 is now size=(15, 27), stride=(1, 1), dtype=int16, device=cuda

# Create an index tensor for torch.gather
# We want to gather along dim 0. Index must be int64.
# Shape of index should be compatible with input for dim 0.
# Let's pick a shape (5, 27) to gather 5 rows.
index = torch.randint(0, 15, (5, 27), dtype=torch.int64, device=device)

# Define the function using torch.gather
def gather_test(input_tensor, index_tensor):
    return torch.gather(input_tensor, dim=0, index=index_tensor)

# Test Eager execution
print("Testing Eager execution...")
try:
    result_eager = gather_test(var_node_1, index)
    print(f"Eager result shape: {result_eager.shape}")
except Exception as e:
    print(f"Eager failed: {e}")
    result_eager = None

# Test Compiled execution
print("Testing Compiled execution...")
try:
    compiled_gather_test = torch.compile(gather_test)
    result_compiled = compiled_gather_test(var_node_1, index)
    print(f"Compiled result shape: {result_compiled.shape}")

    if result_eager is not None:
        if torch.equal(result_eager, result_compiled):
            print("SUCCESS: Eager and Compiled results match.")
        else:
            print("FAILURE: Divergence detected between Eager and Compiled.")
except Exception as e:
    print(f"Compiled failed: {e}")