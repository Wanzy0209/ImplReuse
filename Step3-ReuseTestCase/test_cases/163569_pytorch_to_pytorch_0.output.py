import torch
import torch.nn.functional as F

# Check for CUDA availability as the bug report uses 'cuda' device
if not torch.cuda.is_available():
    print("CUDA is not available. Skipping test.")
    exit()

# Configuration from the bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def test_conv1d_eager_compile_divergence():
    # Inputs based on the bug report
    # arg0: size=(2, 261, 17, 358), dtype=bfloat16, device=cuda
    arg0 = torch.rand([2, 261, 17, 358], dtype=torch.bfloat16, device='cuda', requires_grad=True)
    
    # arg1: size=(17, 64, 358), dtype=float32, device=cuda
    arg1 = torch.rand([17, 64, 358], dtype=torch.float32, device='cuda', requires_grad=True)
    
    # arg2: size=(261, 1, 64), dtype=float32, device=cuda
    arg2 = torch.rand([261, 1, 64], dtype=torch.float32, device='cuda', requires_grad=True)

    def foo(arg0, arg1, arg2):
        t0 = arg0
        t1 = t0.max(dim=0).values
        t2 = t1.transpose(1, 0)
        t3 = arg1
        t4 = torch.exp(t3)
        t5 = arg2
        t6 = t5.transpose(2, 1)
        
        # Original API Call: torch.nn.functional.conv1d
        # Similar API Call: torch.nn.functional.conv1d
        t7 = F.conv1d(t4, t6, stride=1, padding=0)
        
        t8 = t7.clone()
        t8.zero_()
        t9 = t2 * t7 * t8
        return t9

    # 1. Run in Eager Mode
    print("Running Eager Mode...")
    out_eager = foo(arg0, arg1, arg2)
    out_eager.sum().backward()
    print("Eager Success! ")

    # 2. Run in Compiled Mode
    print("Running Compiled Mode...")
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1, arg2)
    out_compiled.sum().backward()
    print("Compile Success! ")

    # 3. Verify Consistency
    # If the bug is a crash, the script stops before here.
    # If the bug is a divergence, this check will fail.
    assert torch.allclose(out_eager, out_compiled), "Divergence detected between eager and compiled outputs!"
    print("Test Passed: Eager and Compiled outputs match.")

if __name__ == '__main__':
    test_conv1d_eager_compile_divergence()