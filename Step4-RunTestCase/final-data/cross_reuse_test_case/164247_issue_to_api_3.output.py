import torch
import torch.nn as nn

# The similar API identified is torch.nn.GroupNorm.
# This test case adapts the original bug reproduction logic (using torch.compile with fullgraph=True)
# to verify if the similar API (GroupNorm) handles the compilation scenario correctly.

class GroupNormTestModel(nn.Module):
    def __init__(self, num_groups=2, num_channels=64):
        super().__init__()
        # Leveraging the similar API: torch.nn.GroupNorm
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)

    def forward(self, x):
        # GroupNorm expects input shape (N, C, *)
        return self.norm(x)

def test_groupnorm_fullgraph():
    # Check if torch.compile is available (requires PyTorch 2.0+)
    if not hasattr(torch, 'compile'):
        print("Test skipped: torch.compile is not available in this PyTorch version (requires PyTorch 2.0+).")
        return

    # Instantiate the model
    model = GroupNormTestModel()
    
    # Reproduce the compilation logic from the original bug report
    # The original issue involves graph breaks with fullgraph=True
    compiled_model = torch.compile(model, fullgraph=True)
    
    # Create dummy input: Batch=2, Channels=64, Length=128
    input_tensor = torch.randn(2, 64, 128)
    
    # Run the compiled model
    try:
        output = compiled_model(input_tensor)
        
        # Basic assertion to ensure the model ran and produced output of correct shape
        assert output.shape == input_tensor.shape, f"Shape mismatch: {output.shape} != {input_tensor.shape}"
        print("Test passed: GroupNorm works with torch.compile(fullgraph=True)")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise

if __name__ == "__main__":
    test_groupnorm_fullgraph()