import torch
import torch.nn.functional as F
import unittest

class TestMPSLinearRegression(unittest.TestCase):
    @unittest.skipIf(not torch.backends.mps.is_available(), "MPS not available")
    def test_linear_non_contiguous_input(self):
        """
        Regression test for Issue #167154.
        Verifies that torch.nn.functional.linear handles non-contiguous 
        inputs (created via as_strided) correctly on the MPS backend 
        without buffer allocation errors.
        """
        # Setup weights and bias
        weight = torch.rand((768, 768), device="mps", dtype=torch.float32)
        bias = torch.rand((768), device="mps", dtype=torch.float32)

        # Setup non-contiguous input using as_strided
        shape = (5, 499, 768)
        stride = (0, 768, 1)
        storage_offset = 0
        
        # Calculate required storage size based on shape and stride
        numel = storage_offset + sum((shape[i] - 1) * stride[i] for i in range(len(shape))) + 1
        base = torch.arange(numel, dtype=torch.float32, device="mps")
        
        input = torch.as_strided(base, size=shape, stride=stride, storage_offset=storage_offset)

        # This call previously triggered an MPS assertion failure:
        # "failed assertion `[MPSNDArray, initWithBufferImpl:offset:descriptor:isForNDArrayAlias:isUserBuffer:] Error: buffer is not large enough"
        output = F.linear(input, weight, bias)

        # Verify output shape is correct
        self.assertEqual(output.shape, (5, 499, 768))

if __name__ == '__main__':
    unittest.main()