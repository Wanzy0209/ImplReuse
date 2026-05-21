import torch
from torch.library import Library

# Define a custom library and operator to test torch.library.register_autograd
lib = Library("test_lib", "DEF")
lib.define("custom_op(Tensor x) -> Tensor")

# Forward implementation
def forward(x):
    return x * 2.0

# Backward implementation
def backward(ctx, grad):
    return grad * 2.0

# Register the forward implementation
torch.library.impl("test_lib::custom_op", forward)

# Register the backward formula using the similar API
torch.library.register_autograd("test_lib::custom_op", backward)

# Test setup
torch.cuda.manual_seed(42)

# Warmup to initialize device state
dummy = torch.ones(1, device="cuda")
torch.ops.test_lib.custom_op(dummy)

# Eager execution
eager_in = torch.ones(1, device="cuda", requires_grad=True)
eager_out = torch.ops.test_lib.custom_op(eager_in)
eager_in_grad, = torch.autograd.grad(eager_out, eager_in)

# CUDA Graph capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    graph_in = torch.ones(1, device="cuda", requires_grad=True)
    graph_out = torch.ops.test_lib.custom_op(graph_in)
    graph_in_grad, = torch.autograd.grad(graph_out, graph_in)

# Replay graph
g.replay()

# Verification
assert torch.allclose(eager_in_grad, graph_in_grad, rtol=0.0, atol=0.0), "Mismatch in gradient outputs"
print("Eager grad:", eager_in_grad)
print("Graph grad:", graph_in_grad)