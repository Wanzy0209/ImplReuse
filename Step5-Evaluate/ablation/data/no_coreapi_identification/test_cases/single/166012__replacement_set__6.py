import torch
import torch.nn as nn

def test_pad_compile_consistency():
    """
    Test case for Issue 166012: tlparse entries on cache hit and not are inconsistent.
    
    This test verifies that models using padding layers produce consistent results
    between cache miss and cache hit scenarios when using torch.compile.
    
    APIs involved:
    - Similar API: torch.nn.ReplicationPad3d
    - Replacement API: torch.nn.ZeroPad2d (replacing tlparse/torch._functorch.config)
    """
    
    # --- Test with Similar API: torch.nn.ReplicationPad3d ---
    class Model3D(nn.Module):
        def __init__(self):
            super().__init__()
            # Using the Similar API
            self.pad = nn.ReplicationPad3d((1, 1, 1, 1, 1, 1))
            self.conv = nn.Conv3d(3, 3, kernel_size=3)

        def forward(self, x):
            return self.conv(self.pad(x))

    model_3d = Model3D()
    input_3d = torch.randn(2, 3, 8, 8, 8)
    
    # Compile the model
    compiled_model_3d = torch.compile(model_3d)
    
    # Run 1: Cache Miss
    out_3d_miss = compiled_model_3d(input_3d)
    
    # Run 2: Cache Hit
    out_3d_hit = compiled_model_3d(input_3d)
    
    # Assert consistency between cache miss and hit
    assert torch.allclose(out_3d_miss, out_3d_hit), \
        "ReplicationPad3d: Outputs differ between cache miss and hit"

    # --- Test with Replacement API: torch.nn.ZeroPad2d ---
    class Model2D(nn.Module):
        def __init__(self):
            super().__init__()
            # Using the Replacement API
            self.pad = nn.ZeroPad2d((1, 1, 1, 1))
            self.conv = nn.Conv2d(3, 3, kernel_size=3)

        def forward(self, x):
            return self.conv(self.pad(x))

    model_2d = Model2D()
    input_2d = torch.randn(2, 3, 8, 8)
    
    # Compile the model
    compiled_model_2d = torch.compile(model_2d)
    
    # Run 1: Cache Miss
    out_2d_miss = compiled_model_2d(input_2d)
    
    # Run 2: Cache Hit
    out_2d_hit = compiled_model_2d(input_2d)
    
    # Assert consistency between cache miss and hit
    assert torch.allclose(out_2d_miss, out_2d_hit), \
        "ZeroPad2d: Outputs differ between cache miss and hit"

if __name__ == "__main__":
    test_pad_compile_consistency()
    print("Test passed successfully.")