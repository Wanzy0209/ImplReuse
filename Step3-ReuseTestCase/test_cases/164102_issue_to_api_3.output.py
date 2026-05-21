import torch
import unittest
import sys

class TestRmsNormDivergence(unittest.TestCase):
    """
    Test case for Issue 164102: [Fuzzer][Eager/Compile Divergence] 
    cannot determine truth value of Relational.
    
    This test reproduces the logic found in the bug report involving 
    torch.rms_norm and other tensor operations to check for 
    eager/compile divergence.
    """
    
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available, skipping test.")
        
        # Configure Dynamo and Inductor as per the bug report
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._inductor.config.emulate_precision_casts = True

    def test_rms_norm_compile_divergence(self):
        # Setup inputs with specific shapes and dtypes from the bug report
        arg0 = torch.rand([93, 62, 23], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        arg1 = torch.rand([93, 62, 11], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        arg2 = torch.rand([93, 62, 10], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        arg3 = torch.rand([93, 62, 81], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        arg4 = torch.rand([93, 62, 2], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        arg5 = torch.rand([93, 62, 8], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        arg6 = torch.rand([77, 8, 127], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        arg7 = torch.rand([16, 8, 15], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        sentinel = torch.tensor(0.0, dtype=torch.bfloat16, device='cuda')

        def foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, sentinel):
            t0 = arg0
            t1 = arg1
            t2 = arg2
            t3 = arg3
            t4 = arg4
            t5 = torch.cat([t0, t1, t2, t3, t4], dim=2)
            t6 = t5.contiguous()
            t7 = arg5
            t8 = torch.exp(t7)
            # The API under test
            t9 = torch.rms_norm(t8, (62, 8))
            t10 = arg6
            t11 = torch.exp(t10)
            t12 = arg7
            t13 = torch.nn.functional.interpolate(t12, size=(127,), mode='nearest')
            t14 = torch.cat([t11, t13], dim=0)
            t15 = torch.baddbmm(t6, t9, t14)
            output = t15 + sentinel
            return output

        # 1. Run in Eager mode
        try:
            eager_output = foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, sentinel)
        except Exception as e:
            self.fail(f"Eager mode failed with: {e}")

        # 2. Run in Compiled mode (torch.compile)
        try:
            compiled_foo = torch.compile(foo)
            compiled_output = compiled_foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, sentinel)
        except Exception as e:
            # The bug report mentions "cannot determine truth value of Relational"
            # which might manifest as a runtime error during compilation or execution.
            self.fail(f"Compiled mode failed with: {e}")

        # 3. Check for divergence
        # We allow a small tolerance for floating point differences, but the shapes must match
        self.assertEqual(eager_output.shape, compiled_output.shape, 
                         "Output shape mismatch between eager and compiled modes")
        
        # Check values are reasonably close
        # Note: bfloat16 precision might require some tolerance, but divergence usually implies large errors or crashes.
        if torch.allclose(eager_output, compiled_output, rtol=1e-2, atol=1e-2):
            pass # Test passed
        else:
            # If values differ significantly, it indicates a divergence bug
            diff = torch.abs(eager_output - compiled_output).max()
            self.fail(f"Eager/Compile divergence detected. Max diff: {diff}")

if __name__ == '__main__':
    unittest.main()