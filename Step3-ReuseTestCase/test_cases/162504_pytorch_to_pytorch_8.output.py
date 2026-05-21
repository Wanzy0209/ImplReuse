import torch
import torch.nn as nn

# Ensure CUDA is available for the test
assert torch.cuda.is_available(), "CUDA is required for this test"

torch.cuda.manual_seed(42)

# Adapt the function 'fn' to be an nn.Module to be used with DataParallel
class SimpleModule(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(torch.randn(1, device="cuda"))

# Initialize module and wrap with DataParallel
# Using device_ids=[0] to target a specific device for the graph capture
module = SimpleModule().cuda()
dp_module = nn.DataParallel(module, device_ids=[0])

# Initialize device state
dp_module(torch.ones(1, device="cuda"))

# Eager execution
torch.cuda.manual_seed(42)
eager_in = torch.ones(1, device="cuda", requires_grad=True)
eager_out = dp_module(eager_in)
eager_in_grad, = torch.autograd.grad(eager_out, eager_in)

# CUDA Graph execution
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    graph_in = torch.ones(1, device="cuda", requires_grad=True)
    graph_out = dp_module(graph_in)
    graph_in_grad, = torch.autograd.grad(graph_out, graph_in)

torch.cuda.manual_seed(42)
g.replay()

# Verification
# Note: DataParallel is generally not compatible with CUDA Graphs due to dynamic control flow.
# This test checks if the behavior matches or if it fails as expected.
assert torch.allclose(eager_in_grad, graph_in_grad, rtol=0.0, atol=0.0), "Mismatch in gradient outputs"
print(eager_in_grad)
print(graph_in_grad)