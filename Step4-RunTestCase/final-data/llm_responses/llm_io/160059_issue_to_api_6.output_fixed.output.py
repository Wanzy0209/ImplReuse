import torch
import torch.nn.functional as F
from torch import nn

# Fix: Mock torch.compiler.disable if the module is missing (e.g., in PyTorch < 2.0)
# This ensures the test can run in environments where torch.compiler is not available.
if not hasattr(torch, 'compiler'):
    class _DummyCompiler:
        @staticmethod
        def disable(recursive=False):
            """Mock decorator for torch.compiler.disable"""
            def decorator(func):
                return func
            return decorator
    torch.compiler = _DummyCompiler()

class Model(nn.Module):
    @torch.compiler.disable(recursive=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Leveraging the similar API: torch.nn.functional.relu6
        # to replace the custom inverse_sigmoid logic from the original issue.
        return F.relu6(x)

def test_jit_script_with_compiler_disable_and_relu6():
    """
    Test that torch.jit.script works correctly with @torch.compiler.disable
    when using torch.nn.functional.relu6.
    """
    model = Model()
    
    # This should not raise an 'undefined value' error
    # (regression test for issue 160059)
    scripted_model = torch.jit.script(model)
    
    # Verify the scripted model executes correctly
    input_tensor = torch.randn(10)
    output = scripted_model(input_tensor)
    
    # Assertions for correctness of relu6
    assert output.shape == input_tensor.shape
    assert (output >= 0).all()
    assert (output <= 6).all()

if __name__ == "__main__":
    test_jit_script_with_compiler_disable_and_relu6()