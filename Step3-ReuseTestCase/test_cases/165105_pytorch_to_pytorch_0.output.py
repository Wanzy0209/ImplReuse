import torch
import pytest

def test_matmul_1d_2d_float16_cuda():
    """
    Test case for torch.matmul with 1D @ 2D inputs on float16 CUDA tensors.
    Based on Issue 165105: [Fuzzer][Eager/Compile Divergence] Matmul decomposition DDE.
    
    This test verifies the specific decomposition path:
    elif dim_tensor1 == 1 and dim_tensor2 == 2:
        return torch.squeeze(torch.mm(torch.unsqueeze(tensor1, 0), tensor2), 0)
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    dtype = torch.float16

    # Recreate inputs based on the fuzzer trace dimensions
    # var_node_4: size=(14,), stride=(1,), dtype=float16, device=cuda
    tensor1 = torch.full((14,), 1.2255859375, dtype=dtype, device=device)
    
    # var_node_5: size=(14, 416), stride=(416, 1), dtype=float16, device=cuda
    # We use random data to simulate the argument inputs (arg_0, etc.) from the trace
    tensor2 = torch.randn(14, 416, dtype=dtype, device=device)

    # Original API Call: torch.matmul(1D, 2D)
    result = torch.matmul(tensor1, tensor2)

    # Assertions for basic properties
    assert result.shape == (416,), f"Expected shape (416,), got {result.shape}"
    assert result.dtype == dtype, f"Expected dtype {dtype}, got {result.dtype}"
    assert result.device == device, f"Expected device {device}, got {result.device}"

    # Verify against the explicit decomposition logic provided in the bug report
    # Decomposition for 1D @ 2D: squeeze(mm(unsqueeze(tensor1, 0), tensor2), 0)
    tensor1_unsqueezed = torch.unsqueeze(tensor1, 0)
    mm_result = torch.mm(tensor1_unsqueezed, tensor2)
    expected = torch.squeeze(mm_result, 0)

    # Check for equality. Note: float16 operations can have precision nuances, 
    # but for the same operation path (mm vs matmul), they should be bitwise equal 
    # if the decomposition is correct.
    assert torch.equal(result, expected), "Divergence detected between torch.matmul and its decomposition logic"

if __name__ == "__main__":
    test_matmul_1d_2d_float16_cuda()
    print("Test passed.")