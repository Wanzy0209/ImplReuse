import torch
import torch.nn as nn

# Handle import error for older PyTorch versions where flex_attention is not available
try:
    from torch.nn.attention.flex_attention import flex_attention
except ImportError:
    flex_attention = None

class FlexAttentionBugRepro(nn.Module):
    """
    Minimal reproduction of the Inductor failure with custom score_mod.
    Based on Issue ID: 163300.
    """
    def __init__(self, N: int, H: int):
        super().__init__()
        self.H = H
        # Register a buffer to simulate the lookup table in the original bug report.
        # Accessing external buffers in score_mod is a key trigger for the layout error.
        self.register_buffer("bias_table", torch.randn(N, H))

    def _score_mod(self, score, b, h, q, kv):
        # Custom score_mod logic that accesses external buffers.
        # This mimics the 'bias = bt[l2, h]' logic from the issue.
        # Using q and kv as indices into the buffer.
        bias = self.bias_table[q, h]
        return score + bias

    def forward(self, q, k, v):
        return flex_attention(q, k, v, score_mod=self._score_mod)

def test_flex_attention_compile_failure():
    """
    Test case to verify the compilation behavior of flex_attention with custom score_mod.
    """
    if flex_attention is None:
        print("Skipping test: torch.nn.attention.flex_attention not found (requires PyTorch >= 2.3).")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Skipping test: CUDA not available.")
        return

    B, H, N, d = 2, 4, 16, 32
    
    # Initialize model
    model = FlexAttentionBugRepro(N, H).to(device)
    
    # Apply torch.compile
    # The bug report indicates failure with dynamic=False
    try:
        compiled_model = torch.compile(model, dynamic=False)
    except Exception as e:
        print(f"Compilation setup failed: {e}")
        return

    # Create dummy inputs
    q = torch.randn(B, H, N, d, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Run eager mode to ensure logic is sound
    try:
        out_eager = model(q, k, v)
        print("Eager execution successful.")
    except Exception as e:
        print(f"Eager execution failed: {e}")
        return

    # Run compiled mode
    # Expecting AssertionError or NoValidChoicesError based on the bug report
    try:
        out_compiled = compiled_model(q, k, v)
        print("Compiled execution successful (Bug might be fixed).")
        
        # Verify outputs match if compilation succeeds
        assert torch.allclose(out_eager, out_compiled, atol=1e-2), "Outputs mismatch"
        
    except AssertionError as e:
        if "convert FlexibleLayout to FixedLayout first" in str(e):
            print(f"Bug reproduced (AssertionError): {e}")
        else:
            raise
    except Exception as e:
        # Catching generic exception to check for NoValidChoicesError or others
        print(f"Compiled execution failed with {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_flex_attention_compile_failure()