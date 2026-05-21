import torch
import pytest

# Check the similar API as requested by the prompt to leverage it.
# This API checks if Flash SDP is enabled, which might be relevant to the backend state
# when running the fuzzer or the compiled model, even if the bug is in torch.unique.
def check_flash_sdp_state():
    if torch.cuda.is_available():
        return torch.backends.cuda.flash_sdp_enabled()
    return False

def test_unique_matmul_divergence():
    """
    Test case for Issue 164876: Eager/Compile Divergence with torch.unique.
    
    The bug arises when torch.compile attempts to handle the output of torch.unique
    (which has a dynamic/symbolic size) in a subsequent matmul operation with a
    statically shaped tensor. The compiler incorrectly infers the shape constraints,
    leading to a runtime error about size mismatch (u0 vs 18).
    """
    
    # Log the state of the similar API
    flash_sdp_enabled = check_flash_sdp_state()
    print(f"Flash SDP Enabled: {flash_sdp_enabled}")

    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True

    torch.manual_seed(1012969)

    # Define the device (CUDA preferred as per original bug report, fallback to CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fuzzed_program(arg_0, arg_1, sentinel):
        var_node_3 = arg_0
        var_node_4 = arg_1
        var_node_2 = torch.matmul(var_node_3.to(torch.float64), var_node_4.to(torch.float64))
        
        # The core of the issue: torch.unique produces a tensor with a size 
        # that might be treated symbolically by the compiler.
        _inp_unique_wide = torch.arange(1, device=var_node_2.device, dtype=torch.int64)
        _uniq_wide = torch.unique(_inp_unique_wide)
        
        var_node_1 = _uniq_wide.to(var_node_2.dtype)
        var_node_5 = torch.full((1, 18), 0.40330381448978797, dtype=torch.float64, device=device)
        
        # This matmul triggers the shape mismatch error in the compiled graph:
        # "The size of tensor a (u0) must match the size of tensor b (18)"
        var_node_0 = torch.matmul(var_node_1.to(torch.float64), var_node_5.to(torch.float64))
        
        result = var_node_0 * sentinel
        if result.is_complex():
            result = result.real
        return result

    sentinel = torch.tensor(1.0, requires_grad=True)
    
    # Setup inputs using as_strided as in the original report
    arg_0 = torch.as_strided(torch.randn(20, device=device).to(torch.float64), (2, 10), (10, 1))
    arg_1 = torch.as_strided(torch.randn(30, device=device).to(torch.float64), (10, 3), (3, 1))

    args = (arg_0, arg_1, sentinel)

    # 1. Run in Eager mode
    try:
        result_eager = fuzzed_program(*args)
        print(" Eager execution succeeded")
    except Exception as e:
        pytest.fail(f"Eager execution failed unexpectedly: {e}")

    # 2. Run in Compiled mode
    # This is expected to fail with the specific shape mismatch error in the buggy version
    compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
    
    try:
        result_compiled = compiled_program(*args)
        print(" Compile execution succeeded")
        
        # If it succeeds, verify results match
        assert torch.allclose(result_eager, result_compiled), "Eager and Compiled results diverged"
        
    except RuntimeError as e:
        error_msg = str(e)
        # Check for the specific error mentioned in the issue
        if "size of tensor a" in error_msg and "must match the size of tensor b" in error_msg:
            print(f" Compile execution failed with expected divergence error: {error_msg}")
            # In a regression test, we might want to assert this does NOT happen.
            # But for a bug reproduction test, catching it confirms the bug exists.
            # Depending on the test goal (reproduction vs verification), we might re-raise.
            # Assuming this is a reproduction test:
            raise 
        else:
            print(f" Compile execution failed with unexpected error: {error_msg}")
            raise
    except Exception as e:
        print(f" Compile execution failed with unexpected exception type: {type(e).__name__}: {e}")
        raise

if __name__ == "__main__":
    test_unique_matmul_divergence()