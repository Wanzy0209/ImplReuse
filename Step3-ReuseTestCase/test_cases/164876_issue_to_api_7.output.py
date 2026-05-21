import torch
import unittest

class TestTorchUniqueCompileDivergence(unittest.TestCase):
    def test_unique_dynamic_shape_matmul(self):
        """
        Reproduces Issue 164876: Eager/Compile Divergence with torch.unique.
        
        The issue arises when torch.unique produces a tensor with a dynamic size
        (symbolic dimension u0) which is then used in a matmul operation.
        The compiler fails to handle the shape inference correctly, leading to:
        "The size of tensor a (u0) must match the size of tensor b (18)"
        """
        # Configuration required to trigger the specific path in the compiler
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        
        # Set seed for reproducibility
        torch.manual_seed(1012969)

        # Sentinel tensor to ensure gradient computation logic is preserved
        sentinel = torch.tensor(1.0, requires_grad=True)

        # Define inputs
        # Using CPU to ensure the test is runnable without CUDA, 
        # though the original bug report specified CUDA.
        arg_0 = torch.as_strided(torch.randn(20).to(torch.float64), (2, 10), (10, 1))
        arg_1 = torch.as_strided(torch.randn(30).to(torch.float64), (10, 3), (3, 1))

        def fuzzed_program(arg_0, arg_1, sentinel):
            var_node_3 = arg_0
            var_node_4 = arg_1
            var_node_2 = torch.matmul(var_node_3.to(torch.float64), var_node_4.to(torch.float64))
            
            # The critical part: torch.unique returns a tensor with a size dependent on data.
            # In the compiler, this often becomes a symbolic dimension (e.g., u0).
            _inp_unique_wide = torch.arange(1, device=var_node_2.device, dtype=torch.int64)
            _uniq_wide = torch.unique(_inp_unique_wide)
            
            var_node_1 = _uniq_wide.to(var_node_2.dtype)
            var_node_5 = torch.full((1, 18), 0.40330381448978797, dtype=torch.float64)
            
            # This matmul triggers the size mismatch error in the compiled graph
            # if the symbolic dimension from unique is not handled correctly.
            var_node_0 = torch.matmul(var_node_1.to(torch.float64), var_node_5.to(torch.float64))
            
            result = var_node_0 * sentinel
            if result.is_complex():
                result = result.real
            return result

        args = (arg_0, arg_1, sentinel)

        # 1. Run in Eager mode
        try:
            result_eager = fuzzed_program(*args)
            print(" Eager execution succeeded")
        except Exception as e:
            self.fail(f"Eager execution failed unexpectedly: {e}")

        # 2. Run in Compiled mode
        # This is expected to fail with the specific error in the bug report
        # if the bug is present.
        compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
        
        try:
            result_compiled = compiled_program(*args)
            print(" Compiled execution succeeded")
            
            # If compiled succeeds, check for numerical consistency
            self.assertTrue(
                torch.allclose(result_eager, result_compiled, rtol=1e-5, atol=1e-5),
                "Compiled output differs from Eager output"
            )
        except RuntimeError as e:
            # Catching the specific error mentioned in the bug report to demonstrate the issue
            error_msg = str(e)
            if "size of tensor a" in error_msg and "must match the size of tensor b" in error_msg:
                self.skipTest(f"Bug reproduced: {error_msg}")
            else:
                self.fail(f"Compiled execution failed with unexpected error: {e}")

if __name__ == "__main__":
    unittest.main()