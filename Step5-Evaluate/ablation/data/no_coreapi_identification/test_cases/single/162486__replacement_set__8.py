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

    # Replacement: torch.randn -> torch.nn.functional.max_pool3d
    # We use torch.ones to create the input tensor for max_pool3d to avoid using torch.randn.
    # max_pool3d expects 5D input (Batch, Channels, Depth, Height, Width).
    # We generate x and y by pooling dummy tensors.
    dummy_input_x = torch.ones(100, 3, 5, 5, 5)
    x = F.max_pool3d(dummy_input_x, kernel_size=2)

    dummy_input_y = torch.ones(100, 2, 5, 5, 5)
    y = F.max_pool3d(dummy_input_y, kernel_size=2)

    dataset = TestDataset(x, y)

    # This call is the subject of the bug report
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [0.7, 0.2, 0.1],
        generator=torch.Generator().manual_seed(42)
    )

    # Assertion to verify the split logic executed correctly
    assert len(train_dataset) + len(val_dataset) + len(test_dataset) == len(dataset)
    print(f"Device {device} worked.")


if __name__ == "__main__":
    # Test on CPU
    test_random_split_with_default_device('cpu')

    # Test on CUDA if available
    if torch.cuda.is_available():
        test_random_split_with_default_device('cuda')
    else:
        print("CUDA not available, skipping CUDA test.")