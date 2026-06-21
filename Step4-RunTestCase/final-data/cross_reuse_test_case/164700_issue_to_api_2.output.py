import torch
import torch.special

# The bug is specific to CUDA/Triton, so we check for availability.
if torch.cuda.is_available():
    device = "cuda"

    def f(x, y):
        # Reproduce the slicing and concatenation pattern from the bug report
        y2 = torch.cat(
            [
                x[:, 1:],
                y[:, None] + 32 * 2048,
            ],
            dim=1,
        )

        x2 = x[:, 1:, None]
        y3 = y2[:, -1:, None]

        # The problematic broadcast and concatenation logic from the original issue
        # We use float inputs to ensure compatibility with torch.special.logsumexp
        combined = torch.cat([x2, y3], dim=1)
        broadcasted = combined + torch.arange(-2048, 0, device=device, dtype=x.dtype)[None, None, :]

        # Leverage the similar API: torch.special.logsumexp
        # We apply logsumexp to the result of the broadcast operation that triggered the crash
        return torch.special.logsumexp(broadcasted, dim=-1)

    # Inputs adapted to float32 to support logsumexp operations
    x = torch.zeros(1, 32, dtype=torch.float32, device=device)
    y = torch.zeros(1, dtype=torch.float32, device=device)

    # Eager execution
    out_eager = f(x, y)

    # Check if torch.compile is available (introduced in PyTorch 2.0)
    if hasattr(torch, 'compile'):
        # Compiled execution
        compiled_f = torch.compile(f)
        out_compiled = compiled_f(x, y)

        # Assertion to check if the compiled output matches the eager output
        # and implicitly verifies that the compiler did not crash.
        assert torch.allclose(out_eager, out_compiled)
    else:
        print("Test skipped: torch.compile not available (requires PyTorch 2.0+)")
else:
    print("Test skipped: CUDA not available")