import torch
import torch.nn.functional as F

# Reproduce the configuration from the bug report
torch._dynamo.config.capture_scalar_outputs = True

# Check for CUDA availability as the original issue was on CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the function using the similar API: torch.nn.functional.embedding_bag
def test_embedding_bag(indices, weight):
    return F.embedding_bag(indices, weight)

# Setup inputs
# The bug report context involves bfloat16 tensors.
# The assertion failure "assert 'int' in str(indices.get_dtype())" suggests 
# that the indices tensor was not an integer type (likely bfloat16 given the fuzzer context).
# We simulate this by creating bfloat16 indices.
weight = torch.randn(10, 5, device=device)
indices = torch.tensor([[0, 1], [2, 3]], dtype=torch.bfloat16, device=device)

# Test Eager mode
print("Testing Eager mode...")
try:
    out_eager = test_embedding_bag(indices, weight)
    print(f"Eager result: {out_eager}")
except Exception as e:
    print(f"Eager mode error: {e}")

# Test Compiled mode (torch.compile)
# This is where the divergence/assertion is expected based on the bug title
print("\nTesting Compiled mode...")
try:
    compiled_fn = torch.compile(test_embedding_bag)
    out_compiled = compiled_fn(indices, weight)
    print(f"Compiled result: {out_compiled}")
except AssertionError as e:
    print(f"Compiled mode assertion error (reproducing bug): {e}")
except Exception as e:
    print(f"Compiled mode error: {e}")