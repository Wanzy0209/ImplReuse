import torch

# Check if MPS is available, otherwise fallback to CPU to ensure compatibility
if torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

# Create a tensor on the selected device
x = torch.tensor([1.0, 2.0, 3.0], device=device)

# Test 1: Default behavior
print("Input:", x)
v1 = torch.vander(x)
print("Vander (default):", v1)

# Test 2: With N specified
v2 = torch.vander(x, N=4)
print("Vander (N=4):", v2)

# Test 3: With increasing=True
v3 = torch.vander(x, increasing=True)
print("Vander (increasing=True):", v3)

# Test 4: With N and increasing
v4 = torch.vander(x, N=2, increasing=True)
print("Vander (N=2, increasing=True):", v4)

# Assertions to verify the output shapes and basic properties
assert v1.shape == (3, 3), "Shape mismatch for default vander"
assert v2.shape == (3, 4), "Shape mismatch for vander with N=4"
assert v3.shape == (3, 3), "Shape mismatch for vander with increasing=True"
assert v4.shape == (3, 2), "Shape mismatch for vander with N=2 and increasing=True"