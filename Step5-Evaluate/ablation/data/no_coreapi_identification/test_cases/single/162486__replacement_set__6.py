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
    set_default_device(device)

    # Replacing torch.randn with torch.nn.functional.max_pool1d
    # We create dummy inputs to feed into max_pool1d to generate the dataset tensors.
    # max_pool1d expects input shape (N, C, L). We use L=10 and kernel_size=2.
    # This replaces the original data generation logic.
    input_x = torch.ones(100, 3, 10)
    x = F.max_pool1d(input_x, kernel_size=2)

    input_y = torch.ones(100, 2, 10)
    y = F.max_pool1d(input_y, kernel_size=2)

    dataset = TestDataset(x, y)

    # BUG: random_split throws an error when default device is 'cuda'
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [0.7, 0.2, 0.1],
        generator=torch.Generator().manual_seed(42)
    )

    # Assertions to verify the split worked correctly
    assert len(train_dataset) == 70
    assert len(val_dataset) == 20
    assert len(test_dataset) == 10
    
    # Verify tensor shapes are preserved after pooling
    sample_x, sample_y = train_dataset[0]
    assert sample_x.shape == (3, 5)
    assert sample_y.shape == (2, 5)

    print(f"Device {device} worked.")


# Run tests
test_bug(device='cpu')  # works

if torch.cuda.is_available():
    test_bug(device='cuda')  # throws an error in the bug report
else:
    print("CUDA not available, skipping CUDA test.")