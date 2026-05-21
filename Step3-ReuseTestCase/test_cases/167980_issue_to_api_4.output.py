import os
import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# Reproduce the environment settings from the bug report
# The crash occurs on both CPU and CUDA, but we force CPU to match the specific test case
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
device = "cpu"
print(f"Running on device: {device}")

# Helper function translating tf.sequence_mask semantics to PyTorch
# This leverages the similar API logic within the PyTorch context
def sequence_mask(lengths, maxlen=None, dtype=torch.bool):
    """
    PyTorch implementation of tf.sequence_mask.
    Returns a mask tensor representing the first N positions of each cell.
    """
    if maxlen is None:
        maxlen = lengths.max()
    
    # Create a range tensor [0, 1, ..., maxlen-1]
    # Expand it to match the batch size of lengths
    row_vector = torch.arange(0, maxlen, 1, device=lengths.device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    
    # Compare: mask[i, j] = (j < lengths[i])
    mask = row_vector < matrix
    
    return mask.to(dtype)

# Define the model structure from the bug report, integrating the similar API logic
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        # Apply sequence masking logic to the input before passing through layers
        # This tests the stability of the module when using the similar API pattern
        batch_size = x.size(0)
        
        # Generate random sequence lengths for the batch
        # In a real scenario, these would be actual sequence lengths
        lengths = torch.randint(low=100, high=784, size=(batch_size,), device=x.device)
        
        # Generate the mask using the translated API
        mask = sequence_mask(lengths, maxlen=784, dtype=torch.float32)
        
        # Flatten input to apply mask
        x_flat = x.view(batch_size, -1)
        
        # Apply mask (zero out features beyond the sequence length)
        x_masked = x_flat * mask
        
        return self.layers(x_masked)

# Data setup
train_data = datasets.MNIST("data", train=True, download=True, transform=ToTensor())
test_data = datasets.MNIST("data", train=False, download=True, transform=ToTensor())

train_loader = DataLoader(train_data, batch_size=80, shuffle=True)
test_loader = DataLoader(test_data, batch_size=80, shuffle=True)

model = Net().to(device)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr=0.01)

print("=== TRAIN ===")
# Run one training step as per the bug report
for X, y in train_loader:
    X, y = X.to(device), y.to(device)
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    break

print("=== EVAL (CRASH POINT) ===")
# Switch to eval mode
model.eval()
    
# Run evaluation step where the crash occurs
with torch.no_grad():
    for batch, (X, y) in enumerate(test_loader):
        print(f"Processing batch {batch}")
        X, y = X.to(device), y.to(device)
        
        # The crash in the original issue happens during the forward pass in eval mode
        # We wrap this in a try-except to catch potential native crashes if they manifest as Python errors
        # (Note: The original bug reports a kernel restart, which might not be catchable here)
        try:
            pred = model(X)
            print("Batch completed successfully.")
        except RuntimeError as e:
            print(f"RuntimeError caught: {e}")
        
        # Only process one batch to match the minimal reproduction
        break
            
print("Test finished.")