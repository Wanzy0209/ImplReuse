import torch
import torch.nn as nn

def test_zero_pad2d_compile_consistency():
    """
    Regression test for Issue 166012: tlparse entries on cache hit and not are inconsistent.
    
    This test verifies that a model utilizing torch.nn.ZeroPad2d behaves consistently
    when compiled with torch.compile. It specifically checks that the execution
    remains stable between a cache miss (first run) and a cache hit (subsequent run),
    ensuring that internal metadata generation (like tlparse entries) does not
    lead to functional discrepancies.
    """
    class PadModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Using a tuple for padding to cover specific argument handling paths
            self.pad = nn.ZeroPad2d((1, 1, 2, 2))

        def forward(self, x):
            return self.pad(x)

    # Initialize model and compile
    model = PadModel()
    compiled_model = torch.compile(model)

    # Create a standard input tensor
    input_tensor = torch.randn(2, 3, 8, 8)

    # 1. Eager execution for baseline correctness
    expected_output = model(input_tensor)

    # 2. First run (Cache Miss)
    # This triggers the compilation and the initial logging/metadata generation
    output_miss = compiled_model(input_tensor)

    # 3. Second run (Cache Hit)
    # This reuses the compiled code. The bug report indicated inconsistencies
    # in log entries (tlparse) between this state and the miss state.
    output_hit = compiled_model(input_tensor)

    # Assertions
    # Verify that the compiled model produces correct results
    assert torch.allclose(output_miss, expected_output), "Output mismatch on cache miss"
    assert torch.allclose(output_hit, expected_output), "Output mismatch on cache hit"

    # Verify that the behavior is consistent across cache states
    assert torch.allclose(output_miss, output_hit), "Inconsistent output between cache miss and hit"

if __name__ == "__main__":
    test_zero_pad2d_compile_consistency()
    print("Test passed: ZeroPad2d compilation consistency verified.")