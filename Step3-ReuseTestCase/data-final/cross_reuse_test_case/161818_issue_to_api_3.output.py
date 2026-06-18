import torch
from torch import nn

def test_nested_tensor_backward_with_dropout():
    """
    Reproduces the logic of Issue 161818 but replaces nn.Linear with 
    torch.nn.Dropout (the similar API) to test backward pass compatibility
    with torch.nested.narrow operations.
    """
    # Use Dropout as the module (Similar API)
    # Note: Dropout does not have learnable parameters, so input requires_grad=True is needed
    module = nn.Dropout(p=0.5)
    module.train()

    padded = torch.rand(9, 8, requires_grad=True)
    lengths = torch.as_tensor([5, 4])

    # Enable anomaly detection to catch the specific error mentioned in the issue
    with torch.autograd.set_detect_anomaly(True):
        # Forward pass
        out = module(padded)
        
        # The critical sequence from the bug report:
        # narrow -> contiguous -> values -> backward
        nopad = torch.nested.narrow(out, dim=1, start=0, length=lengths, layout=torch.jagged).contiguous().values()
        
        # Backward pass
        # Original issue: NotImplementedError: aten::_is_any_true.default
        nopad.sum().backward()

    # Assertions to verify the operation completed successfully
    assert padded.grad is not None, "Gradient for input tensor should be computed."
    assert padded.grad.shape == padded.shape, "Gradient shape should match input shape."

if __name__ == "__main__":
    test_nested_tensor_backward_with_dropout()
    print("Test passed successfully.")