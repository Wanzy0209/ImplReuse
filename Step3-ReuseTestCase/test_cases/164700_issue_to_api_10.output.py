import torch
from torch.backends import mha

def test_compile_cat_unsqueeze_interleaving():
    """
    Test case for Issue 164700: torch.compile generates illegal triton tl.broadcast_to
    and crashes from particular unsqueeze and torch.cat interleaving.
    
    This test leverages torch.backends.mha.get_fastpath_enabled to check the 
    backend optimization state, as compilation behaviors can be sensitive to 
    such global flags.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    # Leverage the similar API: Check the fastpath status
    # This relates to the issue's context of compilation modes and backend behaviors.
    fastpath_enabled = mha.get_fastpath_enabled()
    print(f"Testing with MHA fastpath enabled: {fastpath_enabled}")

    device = "cuda"

    def f(x, y):
        # Original bug reproduction logic
        y2 = torch.cat(
            [
                x[:, 1:],
                y[:, None] + 32 * 2048,
            ],
            dim=1,
        )

        x2 = x[:, 1:, None]
        y3 = y2[:, -1:, None]

        return (
            torch.cat([x2, y3], dim=1)
            + torch.arange(-2048, 0, device=device)[None, None, :]
        ).reshape(1, 32 * 2048)

    # Prepare inputs
    x = torch.zeros(1, 32, dtype=torch.int64, device=device)
    y = torch.zeros(1, dtype=torch.int32, device=device)

    # 1. Run eager execution (baseline)
    expected = f(x, y)

    # 2. Run compiled execution
    # The bug reported a crash here: 'constexpr_type' object has no attribute 'is_block'
    compiled_f = torch.compile(f)
    actual = compiled_f(x, y)

    # 3. Verify correctness
    assert torch.equal(expected, actual), "Compiled output differs from eager output"
    print("Test passed: torch.compile handled the interleaving correctly.")

if __name__ == "__main__":
    test_compile_cat_unsqueeze_interleaving()