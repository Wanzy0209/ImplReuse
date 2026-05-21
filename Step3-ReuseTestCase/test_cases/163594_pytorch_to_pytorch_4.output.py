import torch
import unittest

class TestTorchAll(unittest.TestCase):
    def test_torch_all_basic(self):
        # Basic functionality check
        self.assertTrue(torch.all(torch.tensor([True, True])))
        self.assertFalse(torch.all(torch.tensor([True, False])))

    def test_torch_all_compile(self):
        # Check torch.all behavior with torch.compile
        # Relevant to the original issue involving torch.compile and potential graph breaks

        def fn(x):
            return torch.all(x)

        compiled_fn = torch.compile(fn)

        # Test with boolean tensor
        t1 = torch.tensor([True, True, True])
        self.assertTrue(compiled_fn(t1).item())

        t2 = torch.tensor([True, False, True])
        self.assertFalse(compiled_fn(t2).item())

        # Test with float tensor (non-zero values are True)
        t3 = torch.tensor([1.5, 2.5, 3.5])
        self.assertTrue(compiled_fn(t3).item())

        # Test with float tensor containing zero
        t4 = torch.tensor([1.5, 0.0, 3.5])
        self.assertFalse(compiled_fn(t4).item())

if __name__ == "__main__":
    unittest.main()