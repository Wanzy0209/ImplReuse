import torch

def test_promote_types(device):
    # Check if the device is available to avoid RuntimeError
    if device == "mps":
        # Check if MPS is available in the current PyTorch build
        if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
            print(f"Skipping promote_types test for device: {device} (MPS not available)")
            return
    elif device == "cuda":
        if not torch.cuda.is_available():
            print(f"Skipping promote_types test for device: {device} (CUDA not available)")
            return

    # Adaptation: Use zero-dimensional tensors as in the original bug report
    x = torch.tensor(3.0, device=device)
    y = torch.tensor(1, device=device)
    
    try:
        # torch.promote_types determines the common dtype for type promotion
        # It accepts dtypes (torch.dtype), not Tensors.
        # Fix: Pass x.dtype and y.dtype instead of x and y.
        output = torch.promote_types(x.dtype, y.dtype)
        print(f"promote_types test succeeds for device: {device}. output: {output}")
    except Exception as e:
        print(f"promote_types test fails for device: {device}: {e}")

test_promote_types(device="cpu")
test_promote_types(device="mps")