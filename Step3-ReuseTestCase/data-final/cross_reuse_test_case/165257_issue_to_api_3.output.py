import torch
import unittest
import sys

# The similar API 'tf.executing_eagerly' is used to check the execution context.
# We leverage this pattern to check if the MPS backend is available and suitable
# for testing, similar to how one might check for eager execution in TensorFlow.
def is_mps_context_available():
    """Checks if the MPS backend is available, analogous to tf.executing_eagerly()."""
    return torch.backends.mps.is_available()

class TestMPSRandomInPlaceOps(unittest.TestCase):
    """
    Test case for Issue #165257: MPS Random in-place operations fail silently 
    on non-contiguous tensors.
    """

    def setUp(self):
        # Define the operations that were reported to fail
        self.ops = [
            ("normal_(0,1)", lambda t: t.normal_(0, 1)),
            ("uniform_(0,1)", lambda t: t.uniform_(0, 1)),
            ("exponential_(1)", lambda t: t.exponential_(1.0)),
            ("bernoulli_(0.5)", lambda t: t.bernoulli_(0.5)),
            ("random_()", lambda t: t.random_()),
            ("random_(10)", lambda t: t.random_(10)),
            ("random_(0,10)", lambda t: t.random_(0, 10)),
        ]

    @unittest.skipIf(not is_mps_context_available(), "MPS backend not available")
    def test_non_contiguous_random_ops_mps(self):
        """
        Tests that in-place random operations correctly modify non-contiguous 
        tensors on the MPS device. This mirrors the logic of checking execution
        context (like tf.executing_eagerly) to ensure the operation behaves as expected.
        """
        device = torch.device('mps')
        
        for name, op_func in self.ops:
            with self.subTest(operation=name):
                # Create a non-contiguous tensor on MPS
                # .T creates a non-contiguous view, .clone() materializes it but keeps it non-contiguous
                t_mps = torch.zeros(50, 50, device=device).T.clone()
                
                # Verify it is non-contiguous
                self.assertFalse(t_mps.is_contiguous(), f"Tensor should be non-contiguous for {name}")

                # Apply the in-place random operation
                op_func(t_mps)

                # Check if the operation actually modified the tensor
                # The bug was that max() would remain 0.0
                max_val = t_mps.max().item()
                
                # We assert that the tensor has been modified (max > 0)
                # Note: For bernoulli, values are 0 or 1, so max should be 1 if populated.
                # For uniform/normal, probability of all zeros is effectively 0.
                self.assertGreater(max_val, 0.0, 
                    f"Operation {name} failed silently on non-contiguous MPS tensor (max={max_val})")

    def test_non_contiguous_random_ops_cpu(self):
        """
        Baseline test to ensure operations work correctly on CPU (non-contiguous).
        """
        device = torch.device('cpu')
        
        for name, op_func in self.ops:
            with self.subTest(operation=name):
                t_cpu = torch.zeros(50, 50, device=device).T.clone()
                self.assertFalse(t_cpu.is_contiguous())
                
                op_func(t_cpu)
                max_val = t_cpu.max().item()
                
                self.assertGreater(max_val, 0.0, 
                    f"Operation {name} failed on non-contiguous CPU tensor")

if __name__ == '__main__':
    unittest.main()