import torch
import torch.nn.functional as F

# Set seed for reproducibility since we are using torch.normal
torch.manual_seed(42)

# Use MPS device as in the original issue
device = 'mps'

# Original: a = torch.zeros(1, device='mps')
# Replacement: torch.zeros -> torch.normal
a = torch.normal(0, 1, (1,), device=device)

# Original: a_clamped = a.clamp(min=0.0)
# Similar API: torch.nn.functional.tanh
# Logic: Apply element-wise operation
a_transformed = F.tanh(a)

# Original: b = torch.zeros(1, device='mps')
# Replacement: torch.zeros -> torch.normal
b = torch.normal(0, 1, (1,), device=device)
print(b)

# Original: c = b.clamp(min=1e-7)
# Similar API: torch.nn.functional.tanh
c = F.tanh(b)
print(c)
assert c.device.type == device
assert c.shape == (1,)

# Original: b = torch.zeros(1, device='mps')
# Replacement: torch.zeros -> torch.normal
b = torch.normal(0, 1, (1,), device=device)
print(b)

# Original: c = b.clamp(min=1e-7, max=None)
# Similar API: torch.nn.functional.tanh
c = F.tanh(b)
print(c)

# Original: b = torch.zeros(1, device='mps')
# Replacement: torch.zeros -> torch.normal
b = torch.normal(0, 1, (1,), device=device)
print(b)

# Original: c = b.clamp(min=1e-7, max=torch.inf)
# Similar API: torch.nn.functional.tanh
c = F.tanh(b)
print(c)

# Original: b = torch.zeros(1, device='mps')
# Replacement: torch.zeros -> torch.normal
b = torch.normal(0, 1, (1,), device=device)
print(b)

# Original: c = b.clamp_min(1e-7)
# Similar API: torch.nn.functional.tanh
c = F.tanh(b)
print(c)