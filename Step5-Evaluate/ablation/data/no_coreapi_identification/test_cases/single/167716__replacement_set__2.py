import torch

# Original: a = torch.tensor([[1., 0, 2], [0, 3, 0]]).to_sparse().requires_grad_()
# Replacement: torch.tensor -> torch.isinf
# Adaptation: Removed .to_sparse() because torch.mm expects dense inputs.
# Added .float() because torch.isinf returns a BoolTensor which does not support gradients.
a = torch.isinf([[1., 0, 2], [0, 3, 0]]).float().requires_grad_()

# Original: b = torch.tensor([[0, 1.], [2, 0], [0, 0]], requires_grad=True)
# Replacement: torch.tensor -> torch.isinf
# Adaptation: Added .float() for type consistency.
b = torch.isinf([[0, 1.], [2, 0], [0, 0]]).float().requires_grad=True

# Original: y = torch.sparse.mm(a, b)
# Replacement: torch.sparse.mm -> torch.mm (Similar API)
y = torch.mm(a, b)

# Original: z = y.to_dense()
z = y.to_dense()

# Assertions to verify the test case runs and produces expected output
# Since inputs contain no infinities, isinf returns False (0.0), so result is zero matrix.
assert z.shape == (2, 2)
assert torch.allclose(z, torch.zeros(2, 2))