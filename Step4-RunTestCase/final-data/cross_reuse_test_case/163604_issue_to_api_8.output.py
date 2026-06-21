import torch
import torch.nn.functional as F

# Helper function inspired by tf.compat.v1.train.assert_global_step.
# It validates tensor properties (type, dtype, shape, stride) to ensure
# the test setup matches the specific conditions that trigger the bug.
def assert_tensor_properties(tensor, expected_shape, expected_stride, expected_dtype, name="Tensor"):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} is not a Tensor: {type(tensor)}")
    if tensor.dtype != expected_dtype:
        raise TypeError(f"{name} dtype mismatch. Expected {expected_dtype}, got {tensor.dtype}")
    if tensor.shape != expected_shape:
        raise ValueError(f"{name} shape mismatch. Expected {expected_shape}, got {tensor.shape}")
    if tensor.stride() != expected_stride:
        raise ValueError(f"{name} stride mismatch. Expected {expected_stride}, got {tensor.stride()}")

def test_conv_backward_stride_divergence():
    # Check for torch._dynamo availability (requires PyTorch 2.x)
    if not hasattr(torch, '_dynamo'):
        print("Skipping test: torch._dynamo not found (requires PyTorch 2.x+)")
        return

    # Check for torch._inductor availability
    if not hasattr(torch, '_inductor'):
        print("Skipping test: torch._inductor not found")
        return

    # Configuration from the bug report
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    torch._inductor.config.emulate_precision_casts = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Skipping test: CUDA not available (bug is CUDA specific)")
        return

    # Recreate arg0: size=(4, 503, 64, 504), stride=(16224768, 32256, 504, 1)
    # We use a large storage to accommodate the non-contiguous stride
    storage_size_0 = 16224768 * 4 + 1000
    base_0 = torch.randn(storage_size_0, dtype=torch.float32, device=device)
    arg0 = torch.as_strided(base_0, size=(4, 503, 64, 504), stride=(16224768, 32256, 504, 1))
    arg0.requires_grad = True

    # Recreate arg1: size=(5, 16, 1, 64), stride=(1024, 64, 64, 1)
    storage_size_1 = 1024 * 5 + 1000
    base_1 = torch.randn(storage_size_1, dtype=torch.float32, device=device)
    arg1 = torch.as_strided(base_1, size=(5, 16, 1, 64), stride=(1024, 64, 64, 1))
    arg1.requires_grad = True

    def foo(arg0, arg1):
        t0 = arg0
        # Note: Standard mean produces a contiguous tensor. 
        # To reproduce the bug state (stride=(32192, 64, 1)), we manually adjust the stride
        # to match the fuzzer's observation that triggered the assertion.
        t1 = t0.mean(dim=0)
        
        # Force the specific non-contiguous stride reported in the bug
        # size=(503, 64, 504), stride=(32192, 64, 1)
        # This breaks the autograd graph from t0, so we enable requires_grad manually
        # to test the backward pass of the convolution itself.
        t1 = torch.as_strided(t1, size=(503, 64, 504), stride=(32192, 64, 1))
        t1.requires_grad = True
        
        t2 = F.relu(t1)
        
        t3 = arg1
        t4 = t3.sum(dim=0)
        
        # Force stride for t4: size=(16, 1, 64), stride=(1024, 1, 64)
        t4 = torch.as_strided(t4, size=(16, 1, 64), stride=(1024, 1, 64))
        t4.requires_grad = True
        
        t5 = t4.transpose(2, 1) # size=(16, 64, 1), stride=(1024, 64, 1)
        
        # Leverage the similar API pattern to validate tensor states before the crash point
        assert_tensor_properties(t2, (503, 64, 504), (32192, 64, 1), torch.float32, "Input_Tensor")
        assert_tensor_properties(t5, (16, 64, 1), (1024, 64, 1), torch.float32, "Weight_Tensor")

        t6 = F.conv1d(t2, t5, stride=1, padding=0)
        return t6

    # Test Eager
    try:
        out_eager = foo(arg0, arg1)
        out_eager.sum().backward()
        print("Eager execution passed.")
    except Exception as e:
        print(f"Eager execution failed: {e}")

    # Test Compiled
    # The bug manifests as an assertion error in the compiled backward pass
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    try:
        out_compiled = compiled_foo(arg0, arg1)
        out_compiled.sum().backward()
        print("Compiled execution passed.")
    except Exception as e:
        print(f"Compiled execution failed (Bug Reproduced): {e}")
        # In a regression test, we would assert that this error does NOT occur.
        # Here we output it to verify the reproduction logic.

if __name__ == "__main__":
    test_conv_backward_stride_divergence()