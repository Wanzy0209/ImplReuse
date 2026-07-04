import torch
import sys
import unittest

class TestVarBfloat16CompileDivergence(unittest.TestCase):
    """
    Test case for Issue 164063: TypeError('unexpected type fp32') 
    when using torch.var with bfloat16 under torch.compile with 
    emulate_precision_casts enabled.
    """

    def setUp(self):
        # Check if torch._dynamo is available
        if not hasattr(torch, '_dynamo'):
            self.skipTest("torch._dynamo is not available in this PyTorch build")
        
        # Check if torch._inductor is available
        if not hasattr(torch, '_inductor'):
            self.skipTest("torch._inductor is not available in this PyTorch build")

        # Configuration from the bug report
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._inductor.config.emulate_precision_casts = True

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_var_bfloat16_with_compile(self):
        device = 'cuda'

        # Initialize inputs as per the bug report
        arg0 = torch.rand([36, 7112, 1, 1], dtype=torch.bfloat16, device=device, requires_grad=True)
        arg1 = torch.randint(0, 512, [30, 24], dtype=torch.int64, device=device)
        arg2 = torch.rand([512, 127], dtype=torch.bfloat16, device=device, requires_grad=True)
        arg3 = torch.rand([30, 24, 15], dtype=torch.bfloat16, device=device, requires_grad=True)
        arg4 = torch.rand([30, 4, 16, 127], dtype=torch.bfloat16, device=device, requires_grad=True)
        sentinel = torch.tensor(0.0, dtype=torch.bfloat16, device=device, requires_grad=True)

        def foo(arg0, arg1, arg2, arg3, arg4, sentinel):
            t0 = arg0
            t1 = t0.reshape((28, 24, 3, 127))
            # The bug is triggered here with bfloat16 and specific inductor configs
            t2 = t1.var(dim=2) 
            t3 = arg1
            t4 = arg2
            t5 = torch.nn.functional.embedding(torch.clamp(t3, 0, t4.size(0) - 1).to(torch.long), t4)
            t6 = arg3
            t7 = torch.nn.functional.pad(t6, [0, 1], mode='constant', value=0.0)
            t8 = arg4
            t9 = t8.sum(dim=1)
            t10 = torch.baddbmm(t5, t7, t9)
            t11 = torch.cat([t2, t10], dim=0)
            output = t11 + sentinel
            return output

        # 1. Run Eager Mode
        try:
            out_eager = foo(arg0, arg1, arg2, arg3, arg4, sentinel)
            out_eager.sum().backward()
            print('Eager Success! ')
        except Exception as e:
            self.fail(f"Eager mode failed unexpectedly: {e}")

        # 2. Run Compiled Mode
        # This is where the TypeError('unexpected type fp32') divergence occurs
        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        try:
            out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, sentinel)
            out_compiled.sum().backward()
            print('Compile Success! ')
            
            # Verify results match if both succeed
            self.assertTrue(torch.allclose(out_eager, out_compiled, rtol=1e-2, atol=1e-2))
        except TypeError as e:
            if "unexpected type fp32" in str(e):
                self.fail(f"Bug reproduced: {e}")
            else:
                raise
        except Exception as e:
            self.fail(f"Compiled mode failed with unexpected error: {e}")

if __name__ == '__main__':
    unittest.main()