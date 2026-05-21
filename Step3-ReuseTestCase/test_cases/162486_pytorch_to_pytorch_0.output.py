import torch
from torch.utils.data import Dataset, random_split

class TestDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def test_random_split_with_cuda_default_device():
    """
    Test that torch.utils.data.random_split works correctly when the 
    default device is set to 'cuda'.
    """
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        return

    # Store the original default device to restore it later
    original_device = torch.get_default_device()

    try:
        # Set the default device to CUDA (this triggers the bug in the issue)
        torch.set_default_device('cuda')

        # Create dummy data (tensors will be on CUDA by default)
        x = torch.randn(100, 3)
        y = torch.randn(100, 2)
        dataset = TestDataset(x, y)

        # Case 1: Split using fractional lengths
        # This requires random_split to generate indices, which might 
        # inherit the default device and cause issues.
        lengths = [0.7, 0.2, 0.1]
        train, val, test = random_split(dataset, lengths)

        # Verify the split sizes
        assert len(train) == 70, f"Expected train size 70, got {len(train)}"
        assert len(val) == 20, f"Expected val size 20, got {len(val)}"
        assert len(test) == 10, f"Expected test size 10, got {len(test)}"
        
        # Verify total length is preserved
        assert len(train) + len(val) + len(test) == len(dataset)

        # Case 2: Split using an explicit generator (also mentioned in the bug report)
        generator = torch.Generator().manual_seed(42)
        train_g, val_g, test_g = random_split(dataset, lengths, generator=generator)
        
        assert len(train_g) == 70
        assert len(val_g) == 20
        assert len(test_g) == 10

        print("Test passed: random_split works with default device set to CUDA.")

    except Exception as e:
        print(f"Test failed with error: {e}")
        raise
    finally:
        # Restore the original default device
        torch.set_default_device(original_device)

if __name__ == "__main__":
    test_random_split_with_cuda_default_device()