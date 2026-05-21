import torch
import collections

def test_torch_all_compile():
    """
    Test case for torch.all inside torch.compile.
    This is adapted from the context of the defaultdict regression bug (Issue 166238).
    While collections.defaultdict creation currently fails in torch.compile,
    torch.all should work correctly.
    """
    def fn(x):
        # Use torch.all, the similar API identified
        return torch.all(x > 0)

    # Test with positive case
    x = torch.tensor([1.0, 2.0, 3.0])
    compiled_fn = torch.compile(fn)
    res = compiled_fn(x)
    assert res.item() == True

    # Test with negative case
    x = torch.tensor([1.0, -1.0, 3.0])
    res = compiled_fn(x)
    assert res.item() == False

if __name__ == "__main__":
    test_torch_all_compile()
    print("Test passed.")