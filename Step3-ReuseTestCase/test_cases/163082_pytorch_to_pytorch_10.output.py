import torch

# Ensure reproducibility
torch.manual_seed(1337)

@torch.compile()
def celu_compiled(input_tensor, alpha=1.0):
    return torch.nn.functional.celu(input_tensor, alpha=alpha)

def celu_eager(input_tensor, alpha=1.0):
    return torch.nn.functional.celu(input_tensor, alpha=alpha)

# Check for CUDA availability
if torch.cuda.is_available():
    device = 'cuda'
    # Create a tensor with mixed positive and negative values to test both branches of CELU
    # Using float32 as per the original bug report
    input_tensor = torch.tensor([[1.0, -1.0, 0.5, -0.2]], device=device, dtype=torch.float32)

    print("Input vector:", [x.item() for x in input_tensor[0]])

    # Run compiled version
    out_compiled = celu_compiled(input_tensor)
    print("CELU output (compile):", [x.item() for x in out_compiled[0]])

    # Run eager version
    out_eager = celu_eager(input_tensor)
    print("CELU output (without compile):", [x.item() for x in out_eager[0]])

    # Assert that the results are close enough to be considered correct
    # The original bug showed a deviation, so we check for consistency here.
    assert torch.allclose(out_compiled, out_eager, rtol=1e-5, atol=1e-8), \
        "torch.compile produced inconsistent results for torch.nn.functional.celu on CUDA"
else:
    print("CUDA is not available. Skipping test.")