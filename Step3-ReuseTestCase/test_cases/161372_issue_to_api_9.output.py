import torch
import torch.nn.functional as F
import unittest

class TestConvTranspose3dCompileRegression(unittest.TestCase):
    """
    Test case for torch.compile regression with dynamic shapes.
    Based on Issue 161372, adapted for torch.nn.functional.conv_transpose3d.
    
    The original issue describes a regression where torch.compile hits a 
    recompilation limit due to tensor size mismatches (e.g., expected 77, actual 78).
    This test verifies that conv_transpose3d handles dynamic input shapes 
    correctly under torch.compile without hitting such limits.
    """

    def test_conv_transpose3d_dynamic_shapes_compile(self):
        # Define a function using the similar API: conv_transpose3d
        def forward(x, weight):
            return F.conv_transpose3d(x, weight, stride=2, padding=1)

        # Compile the function
        compiled_forward = torch.compile(forward)

        # Initialize weights
        # Input channels: 3, Output channels: 6, Kernel size: 3
        weight = torch.randn(3, 6, 3, 3, 3)

        # Simulate dynamic input sizes.
        # In the original issue, sequence lengths varied (e.g., 77 vs 78).
        # Here we vary the spatial dimensions (Depth, Height, Width).
        input_shapes = [
            (1, 3, 10, 10, 10),
            (1, 3, 11, 11, 11),
            (1, 3, 12, 12, 12),
            (1, 3, 13, 13, 13),
        ]

        for shape in input_shapes:
            x = torch.randn(*shape)
            
            # Run the compiled function. 
            # If the regression exists, this might hit config.recompile_limit.
            output = compiled_forward(x, weight)

            # Verify output shape calculation
            # Output = (Input - 1) * stride - 2*padding + dilation*(kernel-1) + output_padding + 1
            # With stride=2, padding=1, dilation=1, kernel=3:
            # Output_Dim = (Dim - 1)*2 - 2 + 2 + 1 = 2*Dim - 1
            expected_spatial_dim = 2 * shape[2] - 1
            expected_shape = (shape[0], 6, expected_spatial_dim, expected_spatial_dim, expected_spatial_dim)
            
            self.assertEqual(output.shape, expected_shape, 
                             f"Output shape mismatch for input {shape}. Got {output.shape}, expected {expected_shape}")

if __name__ == "__main__":
    unittest.main()