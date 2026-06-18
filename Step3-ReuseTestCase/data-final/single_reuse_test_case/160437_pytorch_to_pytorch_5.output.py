import torch


@torch.compile(backend="eager")
def fn(x, i):
    if i == 1:
        torch._dynamo.graph_break()
    # Adapted to use torch.any instead of x + 1
    return torch.any(x)


inp = torch.randn(3)
# Call with i=0 (no break)
out1 = fn(inp, 0)
# Call with i=1 (graph break)
out2 = fn(inp, 1)
# Call with i=2 (no break)
out3 = fn(inp, 2)

# Verify results are consistent
assert out1 == out2
assert out2 == out3