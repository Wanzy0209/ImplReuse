import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import set_default_device
from torch.utils.data import random_split


class TestDataset(Dataset): # basic Dataset for testing
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def test_bug(device: str = 'cuda'):
    if device == 'cuda' and not torch.cuda.is_available():
        print(f"Device {device} not available, skipping test.")
        return

    set_default_device(device) 

    # Replacing torch.randn with torch.nn.functional.hinge_embedding_loss
    # We use torch.ones to create the input tensors required for the loss function.
    # reduction='none' is used to preserve the tensor shape (100, 3) and (100, 2)
    # instead of reducing to a scalar, so they can be used as dataset items.
    
    inp_x = torch.ones(100, 3)
    tgt_x = torch.ones(100, 3)
    x = F.hinge_embedding_loss(inp_x, tgt_x, margin=1.0, reduction='none')

    inp_y = torch.ones(100, 2)
    tgt_y = torch.ones(100, 2)
    y = F.hinge_embedding_loss(inp_y, tgt_y, margin=1.0, reduction='none')

    dataset = TestDataset(x, y)

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
    test_bug(device='cpu') # works
    test_bug(device='cuda') # throws an error in the original bug report