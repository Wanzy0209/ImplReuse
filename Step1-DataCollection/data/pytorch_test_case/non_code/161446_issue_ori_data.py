import torch

def test_index_add(device, dtype):
    x = torch.ones([2, 3], dtype=dtype, device=device)
    dim = 0
    ix = torch.tensor(0, dtype=torch.int64, device=device)
    src = torch.ones([1, 3], dtype=dtype, device=device)
    alpha = 1
    try:
        output = torch.index_add(x, dim, ix, src, alpha=alpha)
        print(f"index_add test succeeds for device: {device} & dtype: {dtype}. output shape: {output.shape}")
    except Exception as e:
        print(f"index_add test fails for device: {device} & dtype: {dtype}. error: {e}")

test_index_add("cpu", torch.float32)  # succeeds
test_index_add("mps", torch.float32)  # succeeds

test_index_add("cpu", torch.int32)  # succeeds
test_index_add("mps", torch.int32)  # succeeds

test_index_add("cpu", torch.int64)  # succeeds
test_index_add("mps", torch.int64)  # fails