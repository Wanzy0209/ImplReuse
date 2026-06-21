import torch

# Check if torch._dynamo is available (introduced in PyTorch 2.0)
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True

    def f(x):
        nz = x.nonzero()
        # Replacing the slicing operation (nz[:-1]) with torch.all
        # to test behavior with unbacked sizes.
        return torch.all(nz)

    out = torch.compile(f, fullgraph=True)(torch.randn(3, 4))
    print(out)
else:
    print("Test skipped: torch._dynamo is not available in this version of PyTorch.")