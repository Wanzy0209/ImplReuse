import torch
import unittest

class TestMulSymBoolFakeTensor(unittest.TestCase):
    def test_mul_symbool_with_tensor(self):
        """
        Test case for Issue 164684.
        Verifies that multiplying a scalar boolean (SymBool) with a FakeTensor
        inside a torch.compile context does not raise a TypeError.
        """
        # Check for torch._dynamo availability to handle environment issues
        if not hasattr(torch, '_dynamo'):
            self.skipTest("torch._dynamo is not available in this environment")

        # Configuration required to trigger the specific path in the bug report
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = True

        # Sentinel tensor to ensure gradient computation
        sentinel = torch.tensor(1.0, requires_grad=True)

        # Input argument: boolean tensor
        arg_0 = torch.randint(0, 2, (1,), dtype=torch.bool) > 0

        def fuzzed_program(arg, sent_tensor):
            # Squeeze to scalar tensor
            var_squeezed = torch.squeeze(arg)
            # Extract to Python scalar (bool)
            scalar_bool = var_squeezed.item()
            
            # The operation causing the bug: SymBool * FakeTensor
            result = scalar_bool * sent_tensor
            
            if result.is_complex():
                result = result.real
            return result

        # 1. Run eager mode to establish baseline
        try:
            result_original = fuzzed_program(arg_0, sentinel)
            print(' eager success')
        except Exception as e:
            self.fail(f"Eager mode failed unexpectedly: {e}")

        # 2. Run compiled mode
        compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
        
        try:
            result_compiled = compiled_program(arg_0, sentinel)
            print(' compile success')
        except TypeError as e:
            if "unsupported operand type(s) for *: 'SymBool' and 'FakeTensor'" in str(e):
                self.fail(f"Bug reproduced: {e}")
            else:
                raise

        # 3. Verify results match
        self.assertTrue(torch.allclose(result_original, result_compiled))

if __name__ == '__main__':
    unittest.main()