import torch

# Test case for torch.isposinf on MPS backend
# Adapted from Issue 167767 (clamp incorrectness)

# Test with zero (from original bug context)
b = torch.zeros(1, device='mps')
print(b)
c = torch.isposinf(b)
print(c)
assert not c.item(), "isposinf should return False for zero"

# Test with small positive number (from original bug context)
b = torch.tensor([1e-7], device='mps')
print(b)
c = torch.isposinf(b)
print(c)
assert not c.item(), "isposinf should return False for 1e-7"

# Test with positive infinity (relevant to isposinf)
b = torch.tensor([float('inf')], device='mps')
print(b)
c = torch.isposinf(b)
print(c)
assert c.item(), "isposinf should return True for positive infinity"

# Test with negative infinity (relevant to isposinf)
b = torch.tensor([-float('inf')], device='mps')
print(b)
c = torch.isposinf(b)
print(c)
assert not c.item(), "isposinf should return False for negative infinity"