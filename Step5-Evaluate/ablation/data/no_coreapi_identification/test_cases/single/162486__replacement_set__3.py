import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
from torch import set_default_device


class TestDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def test_bug(device: str = 'cuda'):
    # Check for CUDA availability if device is 'cuda'
    if device == 'cuda' and not torch.cuda.is_available():
        print(f"Device {device} is not available. Skipping test.")
        return

    set_default_device(device)

    # Replacement: torch.randn -> torch.nn.functional.adaptive_max_pool1d
    # We use torch.empty to create the input tensor for the pooling operation to avoid using torch.randn.
    # Input shape (100, 3, 10) -> adaptive_max_pool1d(..., 1) -> (100, 3, 1) -> squeeze -> (100, 3)
    dummy_x = torch.empty(100, 3, 10)
    x = F.adaptive_max_pool1d(dummy_x, output_size=1).squeeze(-1)

    dummy_y = torch.empty(100, 2, 10)
    y = F.adaptive_max_pool1d(dummy_y, output_size=1).squeeze(-1)

    dataset = TestDataset(x, y)

    try:
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [0.7, 0.2, 0.1],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"Device {device} worked.")
        
        # Assertions to verify the split was successful
        assert len(train_dataset) + len(val_dataset) + len(test_dataset) == len(dataset)
        assert len(train_dataset) == 70
        assert len(val_dataset) == 20
        assert len(test_dataset) == 10
        
    except Exception as e:
        print(f"Device {device} failed with error: {e}")
        raise

if __name__ == "__main__":
    # Test with CPU (should work)
    test_bug(device='cpu')
    
    # Test with CUDA (reproduces the bug if not fixed)
    test_bug(device='cuda')