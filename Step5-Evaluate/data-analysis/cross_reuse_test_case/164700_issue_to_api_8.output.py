import torch
import torch.nn.functional as F

def test_compile_issue_with_hardtanh():
    """
    Test case derived from Issue 164700.
    The original issue describes a crash in torch.compile when handling specific
    interleaving of unsqueeze (via None indexing) and torch.cat.
    
    This test case preserves the original reproduction logic while integrating
    the similar API torch.nn.functional.hardtanh_ to verify if the compilation
    path handles the combination of these operations correctly.
    """
    # The bug is specific to CUDA/Triton code generation
    if not torch.cuda.is_available():
        print("Test skipped: CUDA not available")
        return

    # torch.compile was introduced in PyTorch 2.0
    if not hasattr(torch, 'compile'):
        print("Test skipped: torch.compile not available (requires PyTorch 2.0+)")
        return

    device = "cuda"

    def f(x, y):
        # Original logic: interleaving of slicing, unsqueeze, and cat
        y2 = torch.cat(
            [
                x[:, 1:],
                y[:, None] + 32 * 2048,
            ],
            dim=1,
        )

        x2 = x[:, 1:, None]
        y3 = y2[:, -1:, None]

        # Original logic: complex reshape and arithmetic
        res = (
            torch.cat([x2, y3], dim=1)
            + torch.arange(-2048, 0, device=device)[None, None, :]
        ).reshape(1, 32 * 2048)

        # Integration of similar API: torch.nn.functional.hardtanh_
        # Applying an in-place operation to the result of the reshape
        F.hardtanh_(res)

        return res

    # Setup inputs
    x = torch.zeros(1, 32, dtype=torch.int64, device=device)
    y = torch.zeros(1, dtype=torch.int32, device=device)

    # 1. Run in eager mode to establish baseline
    try:
        eager_result = f(x.clone(), y.clone())
    except Exception as e:
        print(f"Eager mode failed unexpectedly: {e}")
        return

    # 2. Run with torch.compile (the context of the bug)
    try:
        compiled_f = torch.compile(f)
        compiled_result = compiled_f(x, y)
    except Exception as e:
        print(f"Compiled mode failed (Bug reproduced): {e}")
        raise

    # 3. Verify results match
    assert torch.equal(eager_result, compiled_result), \
        "Mismatch between eager and compiled outputs"
    
    print("Test passed.")

if __name__ == "__main__":
    test_compile_issue_with_hardtanh()