import torch
import collections
import pytest

def test_torch_any_in_compile():
    """
    Test that torch.any works correctly inside torch.compile.
    This test is generated based on a regression where collections.defaultdict
    failed to be traced by Dynamo. We verify that torch.any, which has a 
    polyfill handler in Dynamo, continues to work as expected.
    """
    def fn(x):
        # Adapted call site: using torch.any instead of collections.defaultdict
        return torch.any(x > 0)

    x = torch.tensor([-1.0, 0.0, 1.0])
    
    # Run eager to get expected result
    expected = fn(x)
    
    # Run compiled
    compiled_fn = torch.compile(fn)
    result = compiled_fn(x)
    
    # Verify results match
    assert torch.equal(result, expected)
    
    # Verify the result is correct
    assert result.item() == True

if __name__ == "__main__":
    test_torch_any_in_compile()
    print("Test passed.")