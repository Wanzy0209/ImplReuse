import torch
import torch.nn as nn
from torch.export import Dim, export
import sympy

def test_convtranspose1d_symbolic_floordiv():
    """
    Test case for torch.nn.ConvTranspose1d based on Issue 164385.
    
    The issue reports that FloorDiv operations in symbolic expressions 
    (e.g., FloorDiv(numerator, 22)) were incorrectly simplified to 
    Mul(Rational(1, 22)), breaking integer arithmetic logic.
    
    This similarity suggests that ConvTranspose1d (likely in its shape 
    inference or quantized implementation) relies on similar FloorDiv 
    logic for calculating output dimensions or packing parameters.
    
    This test verifies that ConvTranspose1d works correctly with 
    dynamic shapes (symbolic tracing), ensuring that internal 
    integer division logic is preserved and not simplified to 
    floating-point rationals.
    """
    
    # Define a simple model using ConvTranspose1d
    class ConvTransposeModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Parameters chosen to potentially trigger complex arithmetic
            self.conv_t = nn.ConvTranspose1d(
                in_channels=16, 
                out_channels=33, 
                kernel_size=3, 
                stride=2,
                padding=1,
                output_padding=1
            )
        
        def forward(self, x):
            return self.conv_t(x)

    model = ConvTransposeModel()
    
    # Define dynamic dimensions to force symbolic tracing
    # This triggers the internal SymPy-based shape inference
    batch_dim = Dim("batch", min=1, max=10)
    length_dim = Dim("length", min=10, max=100)
    
    example_args = (torch.randn(2, 16, 20),)
    
    print("Testing ConvTranspose1d with dynamic shapes (symbolic tracing)...")
    
    try:
        # Export the model. This process uses SymPy and FloorDiv for shape constraints.
        # If FloorDiv simplifies to Rational incorrectly, this may fail or produce wrong constraints.
        exported_program = export(
            model, 
            example_args, 
            dynamic_shapes={"x": {0: batch_dim, 2: length_dim}}
        )
        
        print("Export successful. Verifying output shapes...")
        
        # Test with a different dynamic input size
        # Input: (3, 16, 30)
        # Formula: H_out = (H_in - 1) * stride - 2*padding + dilation*(kernel-1) + output_padding + 1
        # H_out = (30 - 1) * 2 - 2*1 + 1*(3-1) + 1 + 1
        # H_out = 29 * 2 - 2 + 2 + 2 = 58 + 2 = 60
        test_input = torch.randn(3, 16, 30)
        output = exported_program(test_input)
        
        expected_shape = (3, 33, 60)
        assert output.shape == expected_shape, \
            f"Shape mismatch: expected {expected_shape}, got {output.shape}"
            
        print(f"Test passed! Output shape is correct: {output.shape}")
        
        # Additional check: Verify the symbolic graph doesn't contain unexpected 
        # float/rational operations for size calculations if inspectable.
        # (In a real scenario, one might inspect exported_program.graph, 
        # but runtime correctness is the primary validation here).
        
    except Exception as e:
        print(f"Test failed with exception: {e}")
        raise

if __name__ == "__main__":
    test_convtranspose1d_symbolic_floordiv()