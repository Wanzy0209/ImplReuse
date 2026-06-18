import torch

def test_torch_exp_compile():
    # Check for CUDA availability as the original issue was specific to CUDA
    if not torch.cuda.is_available():
        return

    torch.set_default_device('cuda')

    inp = torch.randn(8192)
    
    # Test the similar API: torch.exp
    # Eager execution
    out_eager = torch.exp(inp)
    
    # Compiled execution
    out_compiled = torch.compile(torch.exp)(inp)
    
    # High precision reference
    out_high = torch.exp(inp.to(torch.float64))

    # Verify that the compiled result matches the eager result within a reasonable tolerance.
    # The bug report indicates that Inductor might use fast math, causing divergence.
    # This assertion checks that the behavior is consistent.
    assert torch.allclose(out_eager, out_compiled, rtol=1e-5, atol=1e-5), \
        f"Compiled output differs from eager output. Max diff: {(out_eager - out_compiled).abs().max()}"

    # Optional: Verify against high precision to ensure general correctness
    # (Keeping the logic from the original report to show the comparison)
    diff_eager = (out_high - out_eager).abs().max()
    diff_compiled = (out_high - out_compiled).abs().max()
    
    # Ensure compiled is not significantly worse than eager
    assert diff_compiled <= diff_eager * 1.1 + 1e-6, \
        "Compiled output is significantly less accurate than eager output"

if __name__ == "__main__":
    test_torch_exp_compile()