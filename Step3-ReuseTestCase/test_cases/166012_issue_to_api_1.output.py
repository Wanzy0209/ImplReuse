import torch
import torch.nn as nn

def test_constantpad2d_compile_cache_consistency():
    """
    Test case to verify that torch.compile handles cache hits and misses
    consistently for torch.nn.ConstantPad2d.
    
    This test reproduces the logic of running a compiled model twice
    (once for cache miss, once for cache hit) to ensure behavior consistency.
    """
    # Define a simple model using the similar API: torch.nn.ConstantPad2d
    class PadModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ConstantPad2d(padding=2, value=3.5)

        def forward(self, x):
            return self.pad(x)

    # Initialize model and input
    model = PadModel()
    input_tensor = torch.randn(1, 2, 4, 4)

    # Get expected output from eager mode
    expected_output = model(input_tensor)

    # Compile the model (Original API Under Test context)
    compiled_model = torch.compile(model)

    # First run: This triggers a cache miss and graph compilation
    output_miss = compiled_model(input_tensor)

    # Second run: This triggers a cache hit
    output_hit = compiled_model(input_tensor)

    # Assertions
    # 1. Verify correctness against eager mode
    assert torch.allclose(expected_output, output_miss), \
        "Compiled output (cache miss) does not match eager output."

    # 2. Verify consistency between cache miss and cache hit
    assert torch.allclose(output_miss, output_hit), \
        "Compiled output (cache hit) does not match compiled output (cache miss)."

    print("Test passed: ConstantPad2d compile cache consistency verified.")

if __name__ == "__main__":
    test_constantpad2d_compile_cache_consistency()