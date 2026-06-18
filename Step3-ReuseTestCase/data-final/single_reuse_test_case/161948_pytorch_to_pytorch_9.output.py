import torch

def test_is_floating_point():
    # Adapted from the original context: creating tensors using torch.randn
    # which produces floating point tensors by default.
    tensor_size = (100, 100)
    a = torch.randn(tensor_size)
    b = torch.randn(tensor_size)

    # Test 1: Verify that tensors created by randn are floating point
    assert torch.is_floating_point(a), "Tensor 'a' created by randn should be floating point"
    assert torch.is_floating_point(b), "Tensor 'b' created by randn should be floating point"

    # Test 2: Verify the result of matrix multiplication is floating point
    c = torch.matmul(a, b)
    assert torch.is_floating_point(c), "Result of matmul should be floating point"

    # Test 3: Verify non-floating point tensors return False
    d = torch.ones(tensor_size, dtype=torch.int32)
    assert not torch.is_floating_point(d), "Integer tensor should not be floating point"

    print("All tests passed.")

if __name__ == "__main__":
    test_is_floating_point()