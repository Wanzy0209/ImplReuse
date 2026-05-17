import torch as th

# Setup from the bug report: Create a jagged tensor
x = th.nested.nested_tensor([th.ones(3, 2, 3), th.ones(4, 2, 3)], layout=th.jagged)

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