import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    @torch.compiler.disable(recursive=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Leveraging the similar API (torch.nn.functional.softsign) 
        # inside the decorated forward method to test the regression.
        return F.softsign(x)


def test_jit_script_with_compiler_disable_and_softsign():
    """
    Regression test for Issue 160059.
    Verifies that torch.jit.script does not fail with an 'undefined value' error
    when scripting a module decorated with @torch.compiler.disable, 
    specifically when using torch.nn.functional.softsign.
    """
    model = Model()
    
    # This call should not raise an exception (e.g., 'undefined value')
    # in PyTorch 2.8.0+
    scripted_model = torch.jit.script(model)
    
    # Verify that the scripted model produces the same output as the eager model
    input_tensor = torch.randn(2, 2)
    eager_output = model(input_tensor)
    scripted_output = scripted_model(input_tensor)
    
    assert torch.allclose(eager_output, scripted_output)


if __name__ == "__main__":
    test_jit_script_with_compiler_disable_and_softsign()
    print("Test passed.")