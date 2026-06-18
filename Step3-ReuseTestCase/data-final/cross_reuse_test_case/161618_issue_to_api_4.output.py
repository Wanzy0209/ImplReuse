import torch
from torch.export import export
from torch.export.passes import move_to_device_pass

def test_move_to_device_pass_with_addmm():
    """
    Test case for torch.export.passes.move_to_device_pass based on the 
    logic from Issue #161618 (Inductor/Triton failure on addmm).
    
    This test verifies that the move_to_device_pass correctly handles
    the graph structure and operations (torch.addmm) that were involved
    in the original bug report.
    """
    # Dimensions from the original bug report
    m = 20120
    k = 1536
    n = 512

    # The function that caused issues in the original bug
    def f(a, mat1, mat2):
        return torch.addmm(a, mat1, mat2)

    # Create inputs on CPU to test the "move" functionality
    # (Original bug had them on CUDA, but we test moving the program to CUDA)
    a = torch.randn((m, n))
    mat1 = torch.randn((m, k))
    mat2 = torch.randn((k, n))

    # Export the program
    ep = export(f, args=(a, mat1, mat2))

    # Apply the move_to_device_pass to move the program to CUDA
    # This is the API under test.
    ep_cuda = move_to_device_pass(ep, "cuda")

    # Move inputs to CUDA for execution
    a_cuda = a.cuda()
    mat1_cuda = mat1.cuda()
    mat2_cuda = mat2.cuda()

    # Execute the moved program
    result = ep_cuda(a_cuda, mat1_cuda, mat2_cuda)

    # Verify the result matches the expected output
    expected = torch.addmm(a_cuda, mat1_cuda, mat2_cuda)
    assert torch.allclose(result, expected), "Output mismatch after moving device"

if __name__ == "__main__":
    test_move_to_device_pass_with_addmm()