import torch

# Preserve the original configuration settings from the bug report
# to ensure the test runs under the same compiler conditions.
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

def test_backend_api():
    """
    Test function that leverages the similar API 
    (torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed).
    
    This test checks for Eager/Compile divergence when calling this backend API,
    similar to how the original issue checked torch.unique.
    """
    # Call the similar API
    is_allowed = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
    
    # To ensure the compiler correctly handles the scalar output and control flow,
    # we use the result of the API call to conditionally execute a tensor operation.
    # This mimics the dependency chain in the original issue (unique -> matmul).
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dummy_tensor = torch.randn(2, 2, device=device)
    
    if is_allowed:
        return dummy_tensor + 1.0
    else:
        return dummy_tensor - 1.0

if torch.cuda.is_available():
    # 1. Run in Eager mode
    result_eager = test_backend_api()
    print(' eager success')

    # 2. Run in Compiled mode (fullgraph=True, dynamic=True as in the original issue)
    compiled_program = torch.compile(test_backend_api, fullgraph=True, dynamic=True)
    result_compiled = compiled_program()
    print(' compile success')

    # 3. Assert that there is no divergence
    assert torch.equal(result_eager, result_compiled), \
        f"Divergence detected: Eager result {result_eager} != Compiled result {result_compiled}"
else:
    print("CUDA not available, skipping test for torch.backends.cuda API.")