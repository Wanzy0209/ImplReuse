import torch
import torch.nn as nn

# Set seed for reproducibility
torch.manual_seed(1139008151)

class BuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth = 6
    
    def forward(self, x):
        # Clone, detach, and re-enable gradients
        clone = x.clone()
        detach = clone.detach()
        requires_grad_ = detach.requires_grad_(True)
        
        # Dead code - computed but not used
        # (may trigger compiler optimization issues)
        _ = requires_grad_.ndim
        _ = requires_grad_.ndim + 1
        _ = requires_grad_.ndim >= 1
        
        # Element-wise operations only - should preserve shape
        result = torch.abs(requires_grad_)
        result = torch.sqrt(result)
        result = torch.exp(result)
        result = -result
        result = torch.abs(result)
        result = torch.sqrt(result)
        
        return result

# Create input tensor
x = torch.randn(3, 6, 2)

# Create model
model = BuggyModel().eval()
compiled = torch.compile(model)

# Execute both versions
with torch.no_grad():
    out_eager = model(x)
    out_compiled = compiled(x)

# Compare outputs
print(f"Input shape:     {x.shape}")
print(f"Eager output:    {out_eager.shape}")
print(f"Compiled output: {out_compiled.shape}")
print(f"Shapes match:    {out_eager.shape == out_compiled.shape}")

# This assertion should pass but fails
try:
    torch.testing.assert_close(out_eager, out_compiled)
    print("✓ Outputs match")
except Exception as e:
    print(f"✗ Error: {e}")