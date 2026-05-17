import torch
import unittest

class TestInductorDataDependentSlice(unittest.TestCase):
    def test_compile_data_dependent_slice(self):
        """
        Test case for Issue 161318: Inductor crash on data-dependent slice.
        Verifies that torch.compile with fullgraph=True handles slicing
        based on a scalar derived from tensor operations.
        """
        # Configuration required to trigger the specific path in the bug report
        torch._dynamo.config.capture_scalar_outputs = True

        @torch.compile(fullgraph=True)
        def fn(encoder_attention_mask, encoder_hidden_states):
            # Re-create tensor to match original logic
            encoder_hidden_states = encoder_hidden_states.new_zeros([1, 512, 3072])
            # Calculate scalar from tensor (data-dependent)
            text_len = encoder_attention_mask.sum().item()
            # Slice using the scalar
            encoder_hidden_states = encoder_hidden_states[:, :text_len]
            return encoder_hidden_states

        # Create inputs (using CPU for portability, original used CUDA)
        mask = (torch.arange(512) < 8).unsqueeze(0)
        hidden = torch.randn((1, 512, 4096))

        # Execute and verify no crash occurs
        output = fn(mask, hidden)

        # Verify the slice worked correctly
        self.assertEqual(output.shape, (1, 8, 3072))

if __name__ == "__main__":
    unittest.main()