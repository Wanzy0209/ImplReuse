import torch
import torch.func

def f(a, x):
    # Replaced torch.sparse.mm with torch.mm (the similar API)
    # torch.mm expects dense (strided) tensors
    print(a.layout, x.layout)
    return torch.mm(a, x)

# Replaced torch.tensor with torch.cummin for data generation
# We create base tensors to apply cummin on.
# Note: torch.mm requires dense tensors, so we construct dense inputs.
base_a = torch.tensor([[0., 1., 2.], [1., 2., 0.], [2., 0., 1.]])
base_x = torch.tensor([[1.0], [3.0], [2.0]])

# Using torch.cummin to generate the inputs 'a' and 'x'
# torch.cummin returns (values, indices), we take the values
a, _ = torch.cummin(base_a, dim=0)
x, _ = torch.cummin(base_x, dim=0)

# Test the function
print(f(a, x))

# Test vjp (the context of the original bug)
# For torch.mm with dense tensors, this should work correctly
# unlike the sparse case where layout was not copied.
vjp = torch.func.vjp(f, a, x)[1]
print("VJP successful")