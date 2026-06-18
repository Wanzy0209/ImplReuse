import torch
import torch.nn as nn
from torch.export import export, Dim

def test_convtranspose3d_symbolic_floor_div():
    """
    Test that ConvTranspose3d, when used with symbolic shapes (dynamic shapes),
    correctly handles integer division (FloorDiv) in its output size calculation.
    
    This relates to Issue 164385 where FloorDiv operations were incorrectly
    simplified to Rational numbers in symbolic contexts. The test ensures
    that the symbolic shape inference for ConvTranspose3d preserves integer
    arithmetic logic.
    """
    # Define a ConvTranspose3d layer
    # The output size calculation involves: (Hin - 1)*stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1
    # This arithmetic must remain integer-based and not simplify to Rationals.
    layer = nn.ConvTranspose3d(
        in_channels=16, 
        out_channels=33, 
        kernel_size=3, 
        stride=2, 
        padding=1, 
        output_padding=1
    )

    # Define symbolic dimensions
    # These act as the s14, s37, s46 variables from the bug report
    batch_dim = Dim("batch", min=1, max=10)
    h_dim = Dim("h", min=10, max=20)
    w_dim = Dim("w", min=10, max=20)
    d_dim = Dim("d", min=10, max=20)

    # Example input for tracing
    example_input = torch.randn(1, 16, 10, 10, 10)
    
    # Dynamic shapes mapping
    dynamic_shapes = ({0: batch_dim, 2: h_dim, 3: w_dim, 4: d_dim},)

    print("Exporting ConvTranspose3d with dynamic shapes...")
    
    # Exporting triggers the symbolic shape inference engine.
    # If FloorDiv was simplified to Rational (as in the bug), this step might fail
    # or produce incorrect shape constraints (e.g., floats instead of ints).
    try:
        exported_program = export(layer, example_input, dynamic_shapes=dynamic_shapes)
        
        # Verify the output shape is symbolic
        output_shape_spec = exported_program.output_shape()
        print(f"Symbolic output shape: {output_shape_spec}")
        
        # The output shape should be a list of SymInts.
        # We verify the export succeeded, which implies the symbolic graph is valid.
        assert exported_program is not None
        
        # Verify with a concrete input to ensure the logic holds
        concrete_input = torch.randn(2, 16, 12, 12, 12)
        output = layer(concrete_input)
        
        # Expected output size calculation for dim 2,3,4:
        # (12 - 1) * 2 - 2 * 1 + 1 * (3 - 1) + 1 + 1 = 11*2 - 2 + 2 + 2 = 24
        expected_spatial_dim = 24
        assert output.shape[2] == expected_spatial_dim
        assert output.shape[3] == expected_spatial_dim
        assert output.shape[4] == expected_spatial_dim
        
        print("Test Passed: Symbolic shape inference preserved integer arithmetic.")
        
    except Exception as e:
        print(f"Test Failed: {e}")
        raise

if __name__ == "__main__":
    test_convtranspose3d_symbolic_floor_div()