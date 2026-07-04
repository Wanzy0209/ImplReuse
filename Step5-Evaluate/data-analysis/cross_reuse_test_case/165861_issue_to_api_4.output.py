import torch
import torch.special

# Handle CUDA availability to prevent environment errors
device = "cuda" if torch.cuda.is_available() else "cpu"

# The original bug report highlights a failure when a dimension size is 2**16.
# torch.special.exp2 computes 2^x. We test the boundary around x=16 to ensure
# numerical stability and correctness at this specific value.

# Test the specific boundary value mentioned in the bug (2**16)
# exp2(16) should equal 65536
x = torch.tensor([16.0], device=device)
result = torch.special.exp2(x)
# Fix: Explicitly set dtype to float32 to match the output of exp2, avoiding "Float did not match Long" error
expected = torch.tensor([2**16], device=device, dtype=torch.float32)

assert torch.allclose(result, expected), f"exp2(16) failed: expected {expected.item()}, got {result.item()}"

# Test a value slightly larger to ensure no overflow/precision issues
# similar to the "larger than uint16 max" condition in the bug title
x_large = torch.tensor([17.0], device=device)
result_large = torch.special.exp2(x_large)
# Fix: Explicitly set dtype to float32
expected_large = torch.tensor([2**17], device=device, dtype=torch.float32)

assert torch.allclose(result_large, expected_large), f"exp2(17) failed"

print("exp2 boundary test passed")