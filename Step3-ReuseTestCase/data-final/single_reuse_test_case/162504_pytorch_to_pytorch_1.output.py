import torch

torch.cuda.manual_seed(42)

# Define a list of functions to simulate a sequential model
def fn1(x):
    return x * torch.sigmoid(torch.randn(1, device="cuda"))

def fn2(x):
    return x + 1.0

functions = [fn1, fn2]

# initialize device state
x = torch.ones(1, device="cuda")
for f in functions:
    x = f(x)

torch.cuda.manual_seed(42)
eager_in = torch.ones(1, device="cuda", requires_grad=True)
eager_out = torch.utils.checkpoint.checkpoint_sequential(
    functions, 1, eager_in, # segments=1 implies checkpointing the whole sequence
    use_reentrant=False,
    preserve_rng_state=True,
)
eager_in_grad, = torch.autograd.grad(eager_out, eager_in)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    graph_in = torch.ones(1, device="cuda", requires_grad=True)
    graph_out = torch.utils.checkpoint.checkpoint_sequential(
        functions, 1, graph_in,
        use_reentrant=False,
        preserve_rng_state=True,
    )
    graph_in_grad, = torch.autograd.grad(graph_out, graph_in)

torch.cuda.manual_seed(42)
g.replay()
assert torch.allclose(eager_in_grad, graph_in_grad, rtol=0.0, atol=0.0), "Mismatch in gradient outputs"
print(eager_in_grad)
print(graph_in_grad)