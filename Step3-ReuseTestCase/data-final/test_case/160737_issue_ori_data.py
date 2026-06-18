import torch

def test_index_select(device):
    x = torch.ones([2, 3], device=device)
    index = torch.tensor(1, device=device)  # zero-dimensional index tensor
    try:
        output = torch.index_select(x, dim=0, index=index)
        print(f"index_select test succeeds for device: {device}. output shape: {output.shape}")
    except Exception as e:
        print(f"index_select test fails for device: {device}: {e}")

test_index_select(device = "cpu")
test_index_select(device = "mps")