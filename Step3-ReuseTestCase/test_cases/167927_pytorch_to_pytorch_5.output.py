import torch
import pytest

def test_torch_any_with_compiler_disable_in_fullgraph():
    """
    Test case adapted from Issue 167927.
    Verifies the behavior when torch.any is used inside a torch.compiler.disable
    context, while the outer function is compiled with torch.compile(fullgraph=True).
    
    The original bug report indicates that this scenario raises:
    torch._dynamo.exc.Unsupported: Skip calling `torch.compiler.disable()`d function
    """
    
    def func_with_any(x):
        # Use torch.any inside the disabled block
        with torch.compiler.disable():
            return torch.any(x > 0)

    # Compile with fullgraph=True
    compiled_fn = torch.compile(func_with_any, fullgraph=True)
    
    # Input tensor
    x = torch.tensor([1.0, 2.0, 3.0])

    # Expect the error described in the bug report
    with pytest.raises(torch._dynamo.exc.Unsupported) as excinfo:
        compiled_fn(x)
        
    assert "Skip calling `torch.compiler.disable()`d function" in str(excinfo.value)