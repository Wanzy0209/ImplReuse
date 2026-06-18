import torch
import tempfile
import os

# Setup: Create a temporary file and save a tensor to it
with tempfile.NamedTemporaryFile(delete=False) as f:
    path = f.name

# Save a simple tensor to disk
original_tensor = torch.tensor([0.0, 1.0, 2.0])
torch.save(original_tensor, path)

# Test 1: Basic load
print("Test 1: Basic load")
loaded = torch.load(path)
print(loaded)

# Test 2: Load with map_location='cpu'
print("Test 2: Load with map_location='cpu'")
loaded = torch.load(path, map_location='cpu')
print(loaded)
assert loaded.device.type == 'cpu'

# Test 3: Load with map_location to MPS (if available)
# This mirrors the original bug's context of MPS backend interactions
if torch.backends.mps.is_available():
    print("Test 3: Load with map_location='mps'")
    loaded = torch.load(path, map_location='mps')
    print(loaded)
    assert loaded.device.type == 'mps'

    print("Test 4: Load with map_location=torch.device('mps')")
    loaded = torch.load(path, map_location=torch.device('mps'))
    print(loaded)
    assert loaded.device.type == 'mps'
else:
    print("MPS backend not available, skipping MPS specific tests.")

# Cleanup
os.remove(path)