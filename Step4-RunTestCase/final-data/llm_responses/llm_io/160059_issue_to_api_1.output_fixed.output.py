import torch
from torch import nn

def test_jit_script_with_compiler_disable_and_assert():
    """
    Test case for Issue 160059: torch.jit.script fails with 'undefined value' error
    when module has @torch.compiler.disable decorator.
    
    This test leverages torch._assert (the similar API) to ensure that assertions
    are handled correctly within the context of a disabled compiler during scripting.
    """

    # Check for torch.compiler availability (introduced in PyTorch 2.0)
    # If not available, skip the test as the feature being tested does not exist.
    if not hasattr(torch, 'compiler'):
        print("Skipping test: torch.compiler is not available in this PyTorch version.")
        return

    def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1 / x2)

    class Model(nn.Module):
        @torch.compiler.disable(recursive=False)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = inverse_sigmoid(x)
            
            # Leverage the similar API: torch._assert
            # This checks if the assertion logic interacts correctly with 
            # torch.jit.script when the compiler is disabled.
            torch._assert(x.shape[0] > 0, "Input batch size must be positive")
            
            return x

    n = Model()
    
    # The bug reported in 2.8.0 causes an 'undefined value' error here.
    # We expect this to succeed without raising an exception.
    scripted_model = torch.jit.script(n)
    
    # Verify the scripted model executes correctly
    dummy_input = torch.rand(10)
    output = scripted_model(dummy_input)
    
    assert output is not None
    assert output.shape == dummy_input.shape

if __name__ == "__main__":
    test_jit_script_with_compiler_disable_and_assert()
    print("Test passed.")