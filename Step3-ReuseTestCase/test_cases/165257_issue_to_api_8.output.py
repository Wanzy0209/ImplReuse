import torch
import unittest

# Helper function inspired by tf.ensure_shape to validate tensor properties at runtime.
# This translates the semantic of "ensure_shape" (runtime shape assertion) to 
# "ensure_layout" (runtime contiguity assertion) to verify the bug condition.
def ensure_layout(tensor, expected_contiguous):
    """
    Validates the contiguity of a tensor at runtime.
    Mirrors the behavior of tf.ensure_shape which validates shape at runtime.
    """
    is_contiguous = tensor.is_contiguous()
    if is_contiguous != expected_contiguous:
        raise AssertionError(
            f"Tensor layout assertion failed. Expected contiguous={expected_contiguous}, "
            f"but tensor.is_contiguous() returned {is_contiguous}."
        )
    return tensor

class TestMPSRandomNonContiguous(unittest.TestCase):
    def setUp(self):
        # Skip if MPS is not available
        if not torch.backends.mps.is_available():
            self.skipTest("MPS backend is not available.")

    def test_random_in_place_operations(self):
        """
        Test that random in-place operations modify non-contiguous tensors correctly.
        Bug: On macOS < 15.0, these operations fail silently (tensor remains 0).
        """
        # List of operations to test, matching the original bug report
        ops = [
            ("normal_(0,1)", lambda t: t.normal_(0, 1)),
            ("uniform_(0,1)", lambda t: t.uniform_(0, 1)),
            ("exponential_(1)", lambda t: t.exponential_(1.0)),
            ("bernoulli_(0.5)", lambda t: t.bernoulli_(0.5)),
            ("random_()", lambda t: t.random_()),
            ("random_(10)", lambda t: t.random_(10)),
            ("random_(0,10)", lambda t: t.random_(0, 10)),
        ]

        for name, op_func in ops:
            with self.subTest(operation=name):
                # Create a non-contiguous tensor.
                # Note: We use .T to create a non-contiguous view. 
                # We do NOT use .clone() here as it would make the tensor contiguous,
                # which would bypass the bug condition described in the issue.
                t_mps = torch.zeros(50, 50, device='mps').T

                # Leverage the ensure_shape pattern to verify the pre-condition (non-contiguity)
                ensure_layout(t_mps, expected_contiguous=False)

                # Apply the in-place random operation
                op_func(t_mps)

                # Assert that the tensor was actually modified.
                # The bug causes the tensor to remain all zeros (max == 0).
                self.assertNotEqual(
                    t_mps.max().item(), 
                    0.0, 
                    f"Operation {name} failed silently on non-contiguous tensor (max is 0)"
                )
                
                # Additional check to ensure the tensor isn't just all zeros
                self.assertNotEqual(
                    t_mps.abs().sum().item(), 
                    0.0, 
                    f"Operation {name} failed silently on non-contiguous tensor (sum is 0)"
                )

if __name__ == '__main__':
    unittest.main()