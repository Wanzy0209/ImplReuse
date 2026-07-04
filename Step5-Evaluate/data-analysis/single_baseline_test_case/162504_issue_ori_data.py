# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch

torch.cuda.manual_seed(42)

def fn(x):
    return x * torch.sigmoid(torch.randn(1, device="cuda"))

# initialize device state
fn(torch.ones(1, device="cuda"))

torch.cuda.manual_seed(42)
eager_in = torch.ones(1, device="cuda", requires_grad=True)
eager_out = torch.utils.checkpoint.checkpoint(
    fn, eager_in,
    use_reentrant=False,
    preserve_rng_state=True,
)
eager_in_grad,  = torch.autograd.grad(eager_out, eager_in)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    graph_in = torch.ones(1, device="cuda", requires_grad=True)
    graph_out = torch.utils.checkpoint.checkpoint(
        fn, graph_in,
        use_reentrant=False,
        preserve_rng_state=True,
    )
    graph_in_grad,  = torch.autograd.grad(graph_out, graph_in)

torch.cuda.manual_seed(42)
g.replay()
assert torch.allclose(eager_in_grad, graph_in_grad, rtol=0.0, atol=0.0), "Mismatch in gradient outputs"
print(eager_in_grad)
print(graph_in_grad)