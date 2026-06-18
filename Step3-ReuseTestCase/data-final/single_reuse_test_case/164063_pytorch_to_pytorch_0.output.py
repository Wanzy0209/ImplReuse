import torch
import torch._numpy

# Reproduce the configuration from the bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def test_torch_numpy_var():
    # Setup input: bfloat16 on CUDA
    # Size adapted from bug report (28, 24, 3, 127) -> simplified for minimal test
    x = torch.randn([2, 3, 4, 5], dtype=torch.bfloat16, device='cuda')

    # Function using the Similar API: torch._numpy.var
    # Replacing original t1.var(dim=2) with torch._numpy.var(t1, axis=2)
    def func(x):
        return torch._numpy.var(x, axis=2)

    # Run Eager
    try:
        out_eager = func(x)
        print(f"Eager result shape: {out_eager.shape}, dtype: {out_eager.dtype}")
    except Exception as e:
        print(f"Eager execution failed: {e}")
        return

    # Run Compiled
    try:
        compiled_func = torch.compile(func, fullgraph=True, dynamic=True)
        out_compiled = compiled_func(x)
        print(f"Compiled result shape: {out_compiled.shape}, dtype: {out_compiled.dtype}")

        # Verify consistency
        if torch.allclose(out_eager, out_compiled):
            print("Test Passed: Eager and Compiled outputs match.")
        else:
            print("Test Failed: Outputs differ.")
    except Exception as e:
        print(f"Compiled execution failed: {e}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_torch_numpy_var()
    else:
        print("CUDA not available, skipping test.")