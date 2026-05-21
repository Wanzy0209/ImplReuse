import torch
import unittest

class TestNestedTensorShareMemory(unittest.TestCase):
    def test_nested_tensor_share_memory_jagged_layout(self):
        """
        Test case for Issue #161915: share_memory_() causing segmentation fault.
        
        This test verifies that calling share_memory_() on a NestedTensor 
        with jagged layout completes successfully.
        """
        # Create input tensors
        a = torch.randn(3)
        b = torch.randn(5)
        
        # Initialize NestedTensor
        # This mirrors the initialization pattern of the similar API (tf.keras.layers.InputLayer),
        # which accepts a list of feature columns. Here we pass a list of tensors.
        nt = torch.nested.nested_tensor([a, b], layout=torch.jagged)
        
        # Attempt to share memory. 
        # In the original bug report, this caused a Segmentation Fault.
        # We expect this operation to complete without error and return self.
        try:
            result = nt.share_memory_()
            self.assertIs(result, nt, "share_memory_() should return self (in-place operation)")
        except Exception as e:
            self.fail(f"share_memory_() raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()