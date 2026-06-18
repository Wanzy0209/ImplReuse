import torch
import torch.special

# The bug is specific to CUDA/Triton compilation
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    print("Warning: CUDA not available. Running on CPU, but the bug is specific to CUDA/Triton.")

def f(x, y):
    # Cast to float for torch.special.log1p compatibility
    x_f = x.float()
    y_f = y.float()

    # Replicate the problematic pattern from the bug report:
    # Interleaving of torch.cat, unsqueeze (via None), and broadcasting.
    # We integrate torch.special.log1p into this flow.
    
    y2 = torch.cat(
        [
            x_f[:, 1:],
            # Using torch.special.log1p on the unsqueezed tensor
            torch.special.log1p(y_f[:, None] + 32 * 2048),
        ],
        dim=1,
    )

    x2 = x_f[:, 1:, None]
    y3 = y2[:, -1:, None]

    # Using torch.special.log1p on the result of the second cat
    # before the broadcasting addition with arange.
    return (
        torch.special.log1p(torch.cat([x2, y3], dim=1))
        + torch.arange(-2048, 0, device=device, dtype=torch.float32)[None, None, :]
    ).reshape(1, 32 * 2048)

# Test inputs
x = torch.zeros(1, 32, dtype=torch.float32, device=device)
y = torch.zeros(1, dtype=torch.float32, device=device)

# 1. Run eager mode to establish baseline
try:
    expected = f(x, y)
except Exception as e:
    print(f"Eager mode failed: {e}")
    raise

# 2. Run compiled mode (this is where the crash occurred in the original issue)
try:
    compiled_f = torch.compile(f)
    result = compiled_f(x, y)
except Exception as e:
    print(f"Compiled mode failed: {e}")
    raise

# 3. Verify correctness
assert torch.allclose(expected, result), "Compiled output differs from eager output"
print("Test passed: torch.compile handled torch.special.log1p with the specific cat/unsqueeze pattern.")