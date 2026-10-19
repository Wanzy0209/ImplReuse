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

def test_random_split_with_default_device(device: str = 'cuda'):
    set_default_device(device)

    # Replacing torch.randn with torch.nn.functional.adaptive_max_pool3d
    # To maintain the shape (100, 3) and (100, 2) required by the dataset logic,
    # we create dummy input tensors and apply pooling.
    # torch.empty is used here to create the input tensor for the pooling layer,
    # respecting the set_default_device.
    
    # Create x: target shape (100, 3)
    # Input shape (100, 3, D, H, W) -> Pool to (1, 1, 1) -> View to (100, 3)
    input_x = torch.empty(100, 3, 2, 2, 2)
    x = F.adaptive_max_pool3d(input_x, output_size=(1, 1, 1)).view(100, 3)

    # Create y: target shape (100, 2)
    input_y = torch.empty(100, 2, 2, 2, 2)
    y = F.adaptive_max_pool3d(input_y, output_size=(1, 1, 1)).view(100, 2)

    dataset = TestDataset(x, y)

    # The bug occurs here when default device is 'cuda'
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [0.7, 0.2, 0.1],
        generator=torch.Generator().manual_seed(42)
    )

    # Assertions to verify the split worked correctly
    assert len(train_dataset) == 70, f"Expected train size 70, got {len(train_dataset)}"
    assert len(val_dataset) == 20, f"Expected val size 20, got {len(val_dataset)}"
    assert len(test_dataset) == 10, f"Expected test size 10, got {len(test_dataset)}"
    
    print(f"Device {device} worked.")

if __name__ == "__main__":
    # Test on CPU (should work)
    test_random_split_with_default_device('cpu')

    # Test on CUDA (verifies the bug fix)
    if torch.cuda.is_available():
        test_random_split_with_default_device('cuda')
    else:
        print("CUDA not available, skipping CUDA test.")