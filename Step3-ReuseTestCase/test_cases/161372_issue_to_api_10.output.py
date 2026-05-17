import torch
import torch.nn.functional as F
import unittest

class TestConvTranspose1dCompileRegression(unittest.TestCase):
    def test_conv_transpose1d_dynamic_shape_compile(self):
        """
        Test case derived from Issue 161372 (torch.compile regression).
        
        The original issue reported that torch.compile in 2.8.0 hit the 
        recompile_limit when processing tensors with changing sizes (dynamic shapes),
        specifically in an LLM context involving operations like torch.where.
        
        This test adapts that scenario to the similar API: torch.nn.functional.conv_transpose1d.
        It verifies that compiling a function with conv_transpose1d can handle
        dynamic input lengths without exhausting the recompilation limit.
        """
        
        # Setup parameters for conv_transpose1d
        batch_size = 1
        in_channels = 3
        out_channels = 6
        kernel_size = 3
        
        # Initialize weight (constant for the test)
        weight = torch.randn(in_channels, out_channels, kernel_size)

        # Define the function using the similar API
        def forward(x):
            return F.conv_transpose1d(x, weight)

        # Compile the function
        # Note: In the bug report, the error occurred because the compiler
        # treated different sequence lengths as separate graphs, hitting the limit.
        compiled_forward = torch.compile(forward)

        # Simulate dynamic sequence lengths (e.g., varying token counts in an LLM)
        # The original bug hit the limit at 8 recompilations. We test a range of lengths.
        sequence_lengths = [10, 11, 12, 13, 14, 15, 16, 17, 18]

        try:
            for length in sequence_lengths:
                # Create input with dynamic length dimension
                input_tensor = torch.randn(batch_size, in_channels, length)
                
                # Execute compiled function
                output = compiled_forward(input_tensor)
                
                # Basic assertion to ensure execution
                self.assertEqual(output.shape[0], batch_size)
                self.assertEqual(output.shape[1], out_channels)
                # Output length calculation: L_out = (L_in - 1) * stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1
                # With defaults (stride=1, padding=0, dilation=1, output_padding=0): L_out = L_in + kernel_size - 1
                expected_length = length + kernel_size - 1
                self.assertEqual(output.shape[2], expected_length)
                
        except Exception as e:
            # If the regression exists, this might raise an error related to recompilation limits
            # or graph breaks.
            self.fail(f"torch.compile failed with conv_transpose1d on dynamic shapes: {e}")

if __name__ == "__main__":
    unittest.main()