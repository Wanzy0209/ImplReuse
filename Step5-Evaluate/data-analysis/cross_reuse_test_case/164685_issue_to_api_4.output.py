import torch
import unittest

class TestTorchCompileScalarDivergence(unittest.TestCase):
    """
    Test case for Issue ID: 164685
    Verifies that torch.compile handles scalar division and type conversions
    correctly without raising KeyError: u0, matching eager mode behavior.
    """
    
    def setUp(self):
        # Check if torch._dynamo exists (requires PyTorch 2.0+)
        if not hasattr(torch, '_dynamo'):
            self.skipTest("torch._dynamo is not available in this PyTorch version (requires PyTorch 2.0+)")

        # Configure Dynamo settings as per the bug report
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch.manual_seed(19989)

    def test_compile_divergence_scalar_ops(self):
        """
        Reproduces the eager/compile divergence involving scalar arithmetic.
        """
        def fuzzed_program(arg_0, sentinel):
            var_node_2 = -6  # dtype=int64
            var_node_3 = arg_0  # dtype=int32 (from item())
            var_node_1 = var_node_2 * var_node_3  # dtype=int32
            var_node_5 = torch.full((), 1, dtype=torch.int64)
            var_node_4 = var_node_5.item()  # dtype=int64
            # Division operation causing the divergence in the original issue
            var_node_0 = var_node_1 / var_node_4  # dtype=float (in Python)
            
            # Ensure gradient computation by multiplying with sentinel
            result = var_node_0 * sentinel
            if result.is_complex():
                result = result.real
            return result

        # Sentinel tensor to ensure gradient computation
        sentinel = torch.tensor(1.0, requires_grad=True)

        # Generate input argument matching the fuzzer's logic
        arg_0 = torch.tensor(torch.randn(()), dtype=torch.int32).item()

        # 1. Run in Eager mode
        try:
            result_eager = fuzzed_program(arg_0, sentinel)
            eager_success = True
        except Exception as e:
            eager_success = False
            self.fail(f"Eager mode failed unexpectedly: {e}")

        self.assertTrue(eager_success, "Eager mode should succeed")

        # 2. Run in Compiled mode
        # Using fullgraph=True and dynamic=True as specified in the bug report
        compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
        
        try:
            result_compiled = compiled_program(arg_0, sentinel)
            compile_success = True
        except KeyError as e:
            # Catching the specific error mentioned in the bug report
            self.fail(f"Compiled mode raised KeyError (Bug Reproduction): {e}")
        except Exception as e:
            self.fail(f"Compiled mode failed with unexpected error: {e}")

        self.assertTrue(compile_success, "Compiled mode should succeed without KeyError")

        # 3. Verify results match
        # The bug is a crash, but we should also ensure semantic correctness
        self.assertTrue(
            torch.allclose(result_eager, result_compiled),
            "Compiled output should match eager output"
        )

if __name__ == "__main__":
    unittest.main()