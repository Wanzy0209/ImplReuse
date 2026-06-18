import torch

# Test case for torch.diff
# Based on the extracted implementation details covering arguments: n, axis, prepend, append

def test_torch_diff():
    # 1. Basic usage
    x = torch.tensor([1, 2, 4, 7])
    y = torch.diff(x)
    expected = torch.tensor([1, 2, 3])
    assert torch.equal(y, expected), f"Basic diff failed: {y} != {expected}"

    # 2. Test with n parameter (order of difference)
    x = torch.tensor([1, 2, 4, 8, 16])
    y = torch.diff(x, n=2)
    # 1st diff: [1, 2, 4, 8]
    # 2nd diff: [1, 2, 4]
    expected = torch.tensor([1, 2, 4])
    assert torch.equal(y, expected), f"Diff with n=2 failed: {y} != {expected}"

    # 3. Test with prepend and append
    x = torch.tensor([1, 2, 3])
    prepend_val = torch.tensor([0])
    append_val = torch.tensor([5])
    y = torch.diff(x, prepend=prepend_val, append=append_val)
    # Sequence: [0, 1, 2, 3, 5]
    # Diffs:    [1, 1, 1, 2]
    expected = torch.tensor([1, 1, 1, 2])
    assert torch.equal(y, expected), f"Diff with prepend/append failed: {y} != {expected}"

    # 4. Test with axis parameter
    x = torch.tensor([[1, 3, 5], [2, 4, 6]])
    y = torch.diff(x, axis=0)
    # Diff rows: [2-1, 4-3, 6-5] = [1, 1, 1]
    expected = torch.tensor([[1, 1, 1]])
    assert torch.equal(y, expected), f"Diff with axis=0 failed: {y} != {expected}"

    print("All torch.diff tests passed.")

if __name__ == "__main__":
    test_torch_diff()