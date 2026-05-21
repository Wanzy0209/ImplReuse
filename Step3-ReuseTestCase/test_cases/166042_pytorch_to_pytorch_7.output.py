import torch
import torch.nn.utils

# Replicate the configuration from the bug report
torch._dynamo.config.capture_scalar_outputs = True
torch.manual_seed(1352030645)

# Check for CUDA availability as the original bug occurred on CUDA
if not torch.cuda.is_available():
    print("This test requires a CUDA device.")
else:
    device = 'cuda'
    dtype = torch.bfloat16

    def test_clip_grad_norm(w1, w2):
        # Perform some operations to generate gradients, similar to the fuzzer trace
        x = torch.matmul(w1, w2)
        loss = x.sum()
        loss.backward()
        
        # Call the target API: torch.nn.utils.clip_grad_norm_
        # This replaces the original embedding call site context
        norm = torch.nn.utils.clip_grad_norm_([w1, w2], max_norm=1.0)
        return norm

    # Initialize inputs with bfloat16 on CUDA, matching the bug report's environment
    w1 = torch.randn(4, 8, dtype=dtype, device=device, requires_grad=True)
    w2 = torch.randn(8, 7, dtype=dtype, device=device, requires_grad=True)

    # 1. Run in Eager mode
    w1_eager = w1.clone().detach().requires_grad_()
    w2_eager = w2.clone().detach().requires_grad_()
    eager_result = test_clip_grad_norm(w1_eager, w2_eager)

    # 2. Run in Compiled mode (torch._dynamo)
    compiled_test_clip_grad_norm = torch.compile(test_clip_grad_norm)
    w1_compiled = w1.clone().detach().requires_grad_()
    w2_compiled = w2.clone().detach().requires_grad_()
    compiled_result = compiled_test_clip_grad_norm(w1_compiled, w2_compiled)

    # 3. Check for divergence (Eager vs Compile)
    print(f"Eager Norm: {eager_result}")
    print(f"Compiled Norm: {compiled_result}")
    
    # Assert that the results are close to catch divergence issues
    assert torch.allclose(eager_result, compiled_result), "Divergence detected between eager and compiled execution"