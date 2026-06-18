import torch

# Test case for torch.diag on MPS backend, adapted from the torch.clamp issue.
# This verifies the correctness of torch.diag operations on the MPS device.

# Case 1: Vector to Matrix (default diagonal)
v = torch.tensor([1.0, 2.0, 3.0], device='mps')
print(f"Input Vector: {v}")
m = torch.diag(v)
print(f"Diagonal Matrix: {m}")
expected_m = torch.tensor([[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]], device='mps')
assert torch.equal(m, expected_m), f"Vector to Matrix failed: {m} != {expected_m}"

# Case 2: Matrix to Vector (default diagonal)
m_input = torch.tensor([[1.0, 0.0], [0.0, 2.0]], device='mps')
print(f"Input Matrix: {m_input}")
v_out = torch.diag(m_input)
print(f"Extracted Diagonal: {v_out}")
expected_v = torch.tensor([1., 2.], device='mps')
assert torch.equal(v_out, expected_v), f"Matrix to Vector failed: {v_out} != {expected_v}"

# Case 3: Vector to Matrix with offset (k=1)
v_k = torch.tensor([1.0, 2.0], device='mps')
print(f"Input Vector: {v_k}")
m_k = torch.diag(v_k, k=1)
print(f"Offset Diagonal Matrix: {m_k}")
expected_k = torch.tensor([[0., 1., 0.], [0., 0., 2.], [0., 0., 0.]], device='mps')
assert torch.equal(m_k, expected_k), f"Offset Diagonal failed: {m_k} != {expected_k}"