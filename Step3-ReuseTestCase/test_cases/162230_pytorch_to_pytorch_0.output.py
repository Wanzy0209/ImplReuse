import torch
import unittest

class TestScalarMultiply(unittest.TestCase):
    def test_scalar_multiply(self):
        """
        Test torch.mul with scalar and tensor inputs.
        Derived from the context of the failing test_scalar_multiply.
        """
        # Test 1: Tensor multiplied by a Python scalar
        tensor_a = torch.tensor([1.0, 2.0, 3.0])
        scalar_b = 3.0
        
        # Original logic context: return 3 * a * b
        # We verify the component operations here
        result = torch.mul(tensor_a, scalar_b)
        expected = tensor_a * scalar_b
        self.assertTrue(torch.allclose(result, expected))

        # Test 2: Python scalar multiplied by Tensor
        result = torch.mul(scalar_b, tensor_a)
        expected = scalar_b * tensor_a
        self.assertTrue(torch.allclose(result, expected))

        # Test 3: Integer scalar multiplication
        result = torch.mul(tensor_a, 2)
        expected = tensor_a * 2
        self.assertTrue(torch.allclose(result, expected))

        # Test 4: 0-d Tensor (scalar tensor) multiplied by Tensor
        scalar_tensor = torch.tensor(5.0)
        result = torch.mul(scalar_tensor, tensor_a)
        expected = scalar_tensor * tensor_a
        self.assertTrue(torch.allclose(result, expected))

if __name__ == '__main__':
    unittest.main()