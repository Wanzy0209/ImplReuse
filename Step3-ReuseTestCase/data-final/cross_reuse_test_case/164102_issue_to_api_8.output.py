import torch
import torch._dynamo
import torch._inductor

# Configuration from the bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def test_rms_norm_dynamo_divergence():
    """
    Test case for Issue 164102: [Fuzzer][Eager/Compile Divergence] 
    cannot determine truth value of Relational.
    
    This test preserves the original bug reproduction logic involving torch.rms_norm
    and leverages the similar API torch.backends.cuda.cudnn_sdp_enabled to 
    check the backend state before execution.
    """
    
    if not torch.cuda.is_available():
        print("Test skipped: CUDA not available")
        return

    # Leverage the similar API to check backend state
    # This relates to the issue as the divergence might depend on backend flags
    is_sdp_enabled = torch.backends.cuda.cudnn_sdp_enabled()
    print(f"cuDNN SDP Enabled: {is_sdp_enabled}")

    # Define the function from the issue
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

    # Setup inputs based on the bug report
    arg0 = torch.randn([93, 62, 23], dtype=torch.bfloat16, device='cuda', requires_grad=True)
    arg1 = torch.randn([93, 62, 11], dtype=torch.bfloat16, device='cuda', requires_grad=True)
    arg2 = torch.randn([93, 62, 10], dtype=torch.bfloat16, device='cuda', requires_grad=True)
    arg3 = torch.randn([93, 62, 81], dtype=torch.bfloat16, device='cuda', requires_grad=True)
    arg4 = torch.randn([93, 62, 2], dtype=torch.bfloat16, device='cuda', requires_grad=True)
    arg5 = torch.randn([93, 62, 8], dtype=torch.bfloat16, device='cuda', requires_grad=True)
    arg6 = torch.randn([77, 8, 127], dtype=torch.bfloat16, device='cuda', requires_grad=True)
    arg7 = torch.randn([16, 8, 15], dtype=torch.bfloat16, device='cuda', requires_grad=True)
    sentinel = torch.tensor(0.0, dtype=torch.bfloat16, device='cuda')

    # Run Eager
    try:
        eager_out = foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, sentinel)
    except Exception as e:
        print(f"Eager execution failed: {e}")
        raise

    # Run Compiled
    try:
        compiled_foo = torch._dynamo.optimize("inductor")(foo)
        compiled_out = compiled_foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, sentinel)
    except Exception as e:
        # This is where the bug "cannot determine truth value of Relational" would occur
        print(f"Compiled execution failed: {e}")
        raise

    # Check for divergence
    assert torch.allclose(eager_out, compiled_out, atol=1e-2, rtol=1e-2), "Eager and Compiled outputs diverged"
    print("Test passed: Eager and Compiled outputs match.")

if __name__ == "__main__":
    test_rms_norm_dynamo_divergence()