import torch
import unittest
import sys

# The original bug report indicates a divergence between eager and compiled modes
# specifically on CUDA devices. The similar API provided is tf.test.is_gpu_available.
# We leverage the semantic intent of this API (checking for GPU availability) 
# to guard the test, ensuring it only runs when the necessary hardware is present.

def check_gpu_available():
    """Semantic equivalent of tf.test.is_gpu_available for PyTorch."""
    return torch.cuda.is_available()

class TestTensorAddCompileDivergence(unittest.TestCase):
    
    @unittest.skipIf(not check_gpu_available(), "GPU not available, skipping test.")
    def test_add_zero_sized_tensors_compile(self):
        """
        Test case for Issue 164875.
        Verifies that torch.add on zero-sized tensors (20, 0) behaves consistently
        between eager and torch.compile modes.
        """
        # Configuration from the bug report
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch.manual_seed(1014698)

        def fuzzed_program(arg_0, sentinel):
            # Logic from the original bug report
            var_node_1 = arg_0 
            var_node_3 = torch.full((), True, dtype=torch.bool, device='cuda')
            _x_nz = torch.zeros((), dtype=torch.bool, device=var_node_3.device)
            _x_nz_flat = _x_nz.reshape(-1)
            _x_nz_flat[:20] = True
            var_node_2 = torch.nonzero(_x_nz)
            
            # The operation causing the divergence
            var_node_0 = torch.add(var_node_1, var_node_2)
            
            # Ensure gradient computation
            result = var_node_0 * sentinel
            if result.is_complex():
                result = result.real
            return result

        # Setup inputs
        sentinel = torch.tensor(1.0, requires_grad=True)
        
        # Create the specific strided tensor with size (20, 0) on CUDA
        base_tensor = torch.randint(5, 30, (20,)).to(torch.int64).cuda()
        arg_0 = torch.as_strided(base_tensor, (20, 0), (1, 20))

        # 1. Run in Eager mode
        try:
            result_original = fuzzed_program(arg_0, sentinel)
            print(' eager success')
        except Exception as e:
            self.fail(f"Eager mode failed unexpectedly: {e}")

        # 2. Run in Compiled mode
        compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
        try:
            result_compiled = compiled_program(arg_0, sentinel)
            print(' compile success')
        except RuntimeError as e:
            # Catch the specific error mentioned in the bug report
            if "The size of tensor a" in str(e) and "must match the size of tensor b" in str(e):
                self.fail(f"Compiled mode failed with size mismatch error (Bug 164875): {e}")
            else:
                raise

        # 3. Verify consistency
        self.assertTrue(torch.equal(result_original, result_compiled), 
                        "Divergence detected between eager and compiled results.")

if __name__ == '__main__':
    unittest.main()