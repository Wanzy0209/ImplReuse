import torch

def compute(x, w):
    # Leveraging the similar API pattern (addition) instead of linear
    # to align with the semantic similarity to tf.math.add
    return torch.add(x, w)

def nop(x, w):
    torch._check(x.shape[0] == 0)
    return torch.empty_like(x)

def chunked_compute(x, w):
    sz = x.shape[0]
    torch._check(sz <= 8)
    # Using torch.cond to trigger the control flow graph capture
    out0 = torch.cond(sz > 0, compute, nop, (x[0:2], w))
    out1 = torch.cond(sz > 2, compute, nop, (x[2:4], w))
    out2 = torch.cond(sz > 4, compute, nop, (x[4:6], w))
    out3 = torch.cond(sz > 6, compute, nop, (x[6:8], w))
    return torch.cat([out0, out1, out2, out3])

# Setup inputs
x, w = torch.randn(4, 16, requires_grad=True), torch.randn(16, 16, requires_grad=True)

# Verify eager execution correctness
assert torch.equal(compute(x, w), chunked_compute(x, w))

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(w)

    def forward(self, x):
        return chunked_compute(x, self.w)

# Execute the API under test
# This reproduces the scenario for the "Incorrect user code stack" bug
try:
    mod = torch._dynamo.functional_export._dynamo_graph_capture_for_export(Model())(x)
    print("Test passed: Graph captured successfully.")
except Exception as e:
    print(f"Test failed with error: {e}")
    raise