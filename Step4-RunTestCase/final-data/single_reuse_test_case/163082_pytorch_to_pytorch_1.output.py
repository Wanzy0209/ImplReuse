import torch

torch.manual_seed(1337)

# Fix: Handle environments where torch.compile is not available (PyTorch < 2.0)
# We mock it to act as a pass-through decorator so the test can run without crashing.
if not hasattr(torch, 'compile'):
    print("torch.compile not available. Mocking it as an identity function.")
    torch.compile = lambda *args, **kwargs: lambda func: func

@torch.compile()
def softmax_compiled(input):
    return torch.nn.functional.softmax(input, dim=1)

def softmax_uncompiled(input):
    return torch.nn.functional.softmax(input, dim=1)

device = 'cuda'
# Using the same input as the original bug report to test precision sensitivity
c = torch.tensor([[3.799999, 0.0, 0.0]], device=device, dtype=torch.float32)

print("Input vector:", [x.item() for x in c[0]])

# Run compiled version
out_compiled = softmax_compiled(c)
sum_compiled = torch.sum(out_compiled, dim=1).item()
print("Softmax output (compile):", [x.item() for x in out_compiled[0]], "Sum:", sum_compiled)

# Run uncompiled version
out_uncompiled = softmax_uncompiled(c)
sum_uncompiled = torch.sum(out_uncompiled, dim=1).item()
print("Softmax output (without compile):", [x.item() for x in out_uncompiled[0]], "Sum:", sum_uncompiled)

# Assertions to verify mathematical constraints
# Softmax outputs should sum to 1.0. We check if torch.compile introduces precision errors
# similar to the norm > 1 issue in normalize.
assert torch.isclose(torch.sum(out_compiled, dim=1), torch.tensor(1.0, device=device), atol=1e-6), \
    f"Compiled softmax sum is {sum_compiled}, expected 1.0"
assert torch.isclose(torch.sum(out_uncompiled, dim=1), torch.tensor(1.0, device=device), atol=1e-6), \
    f"Uncompiled softmax sum is {sum_uncompiled}, expected 1.0"