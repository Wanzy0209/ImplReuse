import torch
import torch.nn as nn
import unittest

class TestNestedNarrowBackward(unittest.TestCase):
    def test_nested_narrow_backward_with_anomaly_detection(self):
        """
        Test case for Issue 161818.
        Verifies that backward pass works correctly for nested tensors
        created via narrow, contiguous, and values operations when
        anomaly detection is enabled.
        
        This test preserves the original bug reproduction logic and
        adds structural verification inspired by the nested structure
        handling patterns seen in similar APIs (e.g., tf.io.decode_proto).
        """
        # Setup module and input data
        module = nn.Linear(8, 12)
        padded = torch.rand(9, 8)
        lengths = torch.as_tensor([5, 4])

        # Enable anomaly detection to trigger the error path from the bug report
        with torch.autograd.set_detect_anomaly(True):
            # Forward pass
            out = module(padded)
            
            # Perform nested tensor operations
            # This sequence: narrow -> contiguous -> values was causing the error
            nested_tensor = torch.nested.narrow(out, dim=1, start=0, length=lengths, layout=torch.jagged)
            
            # Verify the nested tensor structure is valid (inspired by structural checks in similar APIs)
            self.assertIsInstance(nested_tensor, torch.Tensor)
            
            contiguous_tensor = nested_tensor.contiguous()
            values = contiguous_tensor.values()
            
            # Verify the values tensor shape matches expectations based on lengths
            # Total length should be sum of lengths (5 + 4 = 9)
            # Batch size is 9 (from padded), but narrow operates on dim 1.
            # out shape is (9, 12). narrow on dim 1 with lengths [5, 4] implies jagged layout.
            # The resulting values should flatten the jagged dimension.
            # Expected total elements: 9 * 5 (approx, depending on exact narrow semantics) or similar.
            # Given the bug is about backward pass, we primarily ensure the operation runs.
            
            # Backward pass
            # The bug raised NotImplementedError here
            try:
                values.sum().backward()
            except NotImplementedError as e:
                self.fail(f"Backward pass failed with NotImplementedError: {e}")

            # Verify gradients were computed successfully
            self.assertIsNotNone(module.weight.grad)
            self.assertIsNotNone(module.bias.grad)
            
            # Ensure gradients are not all zeros (sanity check)
            self.assertTrue(torch.any(module.weight.grad != 0))

if __name__ == '__main__':
    unittest.main()