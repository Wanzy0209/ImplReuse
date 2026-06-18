import torch
import torch.nn as nn

def test_lazy_linear_in_custom_module():
    """
    Test case adapted from Issue 162147 context.
    Replaces torch.nn.Linear with torch.nn.LazyLinear to verify
    initialization and gradient flow in a custom module context.
    """
    # Dimensions from the original bug report context
    D_in, H, D_out = 128, 64, 10
    d_addr = 64
    batch_size = 32

    class CustomModule(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Original API: self.P = nn.Linear(D_in, d_addr, bias=False)
            # Adaptation: Use torch.nn.LazyLinear.
            # Note: LazyLinear infers in_features from the input size during the first forward pass.
            self.P = nn.LazyLinear(d_addr, bias=False)
            
            # Other parameters to mimic the original module structure
            self.W = nn.Parameter(torch.randn(10, H, D_in))

        def forward(self, x):
            # The forward pass triggers the lazy initialization of self.P
            return self.P(x)

    model = CustomModule()
    x = torch.randn(batch_size, D_in)

    # Verify lazy initialization state before forward pass
    assert model.P.weight is None, "LazyLinear weight should be None before forward pass"

    # Forward pass
    output = model(x)

    # Verify initialization happened correctly
    assert model.P.weight is not None, "LazyLinear weight was not initialized"
    # Weight shape for Linear(out, in) -> LazyLinear(out) infers in
    assert model.P.weight.shape == (d_addr, D_in), \
        f"Expected weight shape ({d_addr}, {D_in}), got {model.P.weight.shape}"

    # Verify backward pass (addressing the "training" context of the bug report)
    loss = output.sum()
    loss.backward()

    assert model.P.weight.grad is not None, "Gradients were not computed"
    assert model.P.weight.grad.shape == model.P.weight.shape, "Gradient shape mismatch"

    print("Test Passed: torch.nn.LazyLinear initialized and computed gradients correctly.")

if __name__ == "__main__":
    test_lazy_linear_in_custom_module()