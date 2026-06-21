import torch
import sys

def test_unique_compile_divergence():
    """
    Test case for Issue 164876: Eager/Compile Divergence with torch.unique.
    This test preserves the original bug reproduction logic and leverages
    torch.backends.cuda.cudnn_sdp_enabled to check the backend environment.
    """
    
    # Leverage the similar API to check the CUDA backend state
    # This reflects the relationship by using the similar API in the test setup.
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        return

    # Fix: Handle missing attribute gracefully for different PyTorch versions
    if hasattr(torch.backends.cuda, 'cudnn_sdp_enabled'):
        sdp_enabled = torch.backends.cuda.cudnn_sdp_enabled()
        print(f"cuDNN SDP enabled: {sdp_enabled}")
    else:
        print("cuDNN SDP enabled check skipped (attribute not found in this version)")

    # Configuration from the original bug report
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    torch.manual_seed(1012969)

    # Sentinel tensor to ensure gradient computation
    sentinel = torch.tensor(1.0, requires_grad=True)

    # Input arguments
    arg_0 = torch.as_strided(torch.randn(20).to(torch.float64), (2, 10), (10, 1)).cuda()
    arg_1 = torch.as_strided(torch.randn(30).to(torch.float64), (10, 3), (3, 1)).cuda()

    def fuzzed_program(arg_0, arg_1, sentinel):
        var_node_3 = arg_0
        var_node_4 = arg_1
        var_node_2 = torch.matmul(var_node_3.to(torch.float64), var_node_4.to(torch.float64))
        
        # Original API Under Test: torch.unique
        _inp_unique_wide = torch.arange(1, device=var_node_2.device, dtype=torch.int64)
        _uniq_wide = torch.unique(_inp_unique_wide)
        
        var_node_1 = _uniq_wide.to(var_node_2.dtype)
        var_node_5 = torch.full((1, 18), 0.40330381448978797, dtype=torch.float64, device=var_node_2.device)
        
        # The divergence occurs here due to shape mismatch in compiled mode
        var_node_0 = torch.matmul(var_node_1.to(torch.float64), var_node_5.to(torch.float64))
        
        result = var_node_0 * sentinel
        if result.is_complex():
            result = result.real
        return result

    args = (arg_0, arg_1, sentinel)

    try:
        # Run in Eager mode
        result_eager = fuzzed_program(*args)
        print(' Eager execution success')

        # Run in Compiled mode
        compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
        result_compiled = compiled_program(*args)
        print(' Compile execution success')

        # Check for divergence
        if not torch.allclose(result_eager, result_compiled):
            print(" Divergence detected between eager and compiled results")
            sys.exit(1)
        else:
            print(" Results match")

    except RuntimeError as e:
        print(f" Error during execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_unique_compile_divergence()