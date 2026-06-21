import torch
import torch.nn as nn

# Fix: Handle environments where torch.compile is not available (PyTorch < 2.0)
if not hasattr(torch, 'compile'):
    # Mock torch.compile as a pass-through decorator
    torch.compile = lambda *args, **kwargs: lambda func: func

if torch.cuda.is_available():
    torch.manual_seed(1337)

    @torch.compile()
    def hard_shrink_compiled(x):
        # Using the similar API: torch.nn.Hardshrink
        return nn.Hardshrink()(x)

    def hard_shrink_uncompiled(x):
        return nn.Hardshrink()(x)

    device = 'cuda'
    # Input tensor with values crossing the default lambda=0.5 threshold
    # 1.0 > 0.5 (keep), 0.4 < 0.5 (shrink to 0), -0.6 < -0.5 (keep)
    c = torch.tensor([[1.0, 0.4, -0.6]], device=device, dtype=torch.float32)
    
    print("Input vector:", [x.item() for x in c[0]])
    
    out_c = hard_shrink_compiled(c)
    print("Output (compile):", [x.item() for x in out_c[0]])
    
    out_nc = hard_shrink_uncompiled(c)
    print("Output (without compile):", [x.item() for x in out_nc[0]])

    # Verify consistency between compiled and uncompiled versions
    assert torch.allclose(out_c, out_nc), "Outputs differ between compiled and uncompiled versions"
else:
    print("CUDA is not available. Test skipped.")