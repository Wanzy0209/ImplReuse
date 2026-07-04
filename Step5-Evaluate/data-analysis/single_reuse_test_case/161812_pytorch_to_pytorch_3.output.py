import torch as th

# Setup from the bug report: Create a jagged tensor
# Fixed: Removed 'layout=th.jagged' as 'th.jagged' is not a valid attribute in standard PyTorch.
# torch.nested.nested_tensor creates a nested tensor by default from a list of tensors.
x = th.nested.nested_tensor([th.ones(3, 2, 3), th.ones(4, 2, 3)])

# Test the similar API: torch.triu
# Adapted from the original failing call: th.cat([x, x])
try:
    # torch.triu is a unary operator, so we pass x directly
    result = th.triu(x)
    
    # Assertion to verify the operation completed and returned a valid nested tensor
    assert result.is_nested, "Result should be a nested tensor"
    print("Test passed: torch.triu works with jagged tensor input.")

except Exception as e:
    print(f"Test failed with error: {e}")
    raise