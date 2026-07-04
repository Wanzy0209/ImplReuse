import torch
import sys

# Reproduce the environment configuration from the original bug report
# Guard against older PyTorch versions where _dynamo might not exist
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
else:
    print("Warning: torch._dynamo not found. Skipping specific dynamo configurations.")

torch.manual_seed(70609)

# The original bug was observed on CUDA, but we allow fallback to CPU for runnability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_split_divergence():
    # Check if torch.compile is available
    if not hasattr(torch, 'compile'):
        print("Skipping test: torch.compile is not available in this PyTorch version.")
        return

    # Synthesizing inputs based on the original fuzzer program's tensor properties
    # var_node_9: size=(6, 13), stride=(13, 1), dtype=float16, device=cuda
    var_node_9 = torch.full((6, 13), 1.3154296875, dtype=torch.float16, device=device)
    
    # var_node_10: size=(13, 1), stride=(1, 1), dtype=float16, device=cuda
    # In the original code, this was arg_1. We create a tensor to match the context.
    var_node_10 = torch.full((13, 1), 1.3154296875, dtype=torch.float16, device=device)

    # Original call site:
    # var_node_8 = torch.matmul(var_node_9.to(torch.float16), var_node_10.to(torch.float16))
    
    # Adapted call site using torch.split:
    # We replace the matrix multiplication with a split operation on var_node_9.
    # This tests the similar API (torch.split) with the same tensor characteristics (float16, specific shapes).
    # We split along dimension 0 into chunks of size 2.
    var_node_8 = torch.split(var_node_9.to(torch.float16), 2, dim=0)

    # To verify the API, we check for Eager/Compile Divergence as indicated by the bug title.
    # We define a function that performs the split and run it in both modes.
    
    def run_split(x):
        return torch.split(x, 2, dim=0)

    # Eager execution
    eager_result = run_split(var_node_9)
    
    # Compiled execution (using torch.compile which uses torch._dynamo)
    compiled_fn = torch.compile(run_split)
    compiled_result = compiled_fn(var_node_9)

    # Assertions to verify correctness and check for divergence
    assert len(eager_result) == len(compiled_result), "Number of splits differs between eager and compiled modes"
    
    for i, (e_tensor, c_tensor) in enumerate(zip(eager_result, compiled_result)):
        assert e_tensor.shape == c_tensor.shape, f"Shape mismatch at split index {i}"
        assert torch.equal(e_tensor, c_tensor), f"Value mismatch (divergence) at split index {i}"

if __name__ == "__main__":
    test_split_divergence()