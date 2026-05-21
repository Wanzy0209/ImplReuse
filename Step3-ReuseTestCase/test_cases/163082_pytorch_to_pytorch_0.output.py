import torch

def test_normalize_compile_cuda_fp32():
    """
    Test case to verify that torch.nn.functional.normalize does not output
    vectors with norm > 1 when using torch.compile on CUDA with float32 inputs.
    """
    if not torch.cuda.is_available():
        print("Test skipped: CUDA not available")
        return

    torch.manual_seed(1337)

    @torch.compile()
    def vec_norm(e_dist):
        return torch.nn.functional.normalize(e_dist)

    device = 'cuda'
    # Input from the bug report that triggers the issue
    c = torch.tensor([[3.799999, 0.0, 0.0]], device=device, dtype=torch.float32)

    # Run compiled version
    xyz_compiled = vec_norm(c)

    # Calculate the L2 norm of the result
    norm_val = torch.linalg.norm(xyz_compiled, dim=1).item()

    # The bug causes the norm to be slightly greater than 1.0 (e.g., 1.0000001192092896).
    # We assert that the norm must be <= 1.0 (allowing for extremely minimal floating point noise).
    # The specific bug value is ~1.19e-7 over 1.0, so a tolerance of 1e-7 is appropriate to catch the bug
    # while allowing standard precision.
    assert norm_val <= 1.0 + 1e-7, f"Norm {norm_val} is greater than 1.0"

    # Verify that acos does not return NaN (as mentioned in the bug description)
    # acos(x) is undefined for x > 1
    acos_val = torch.acos(xyz_compiled[0, 0])
    assert not torch.isnan(acos_val), "acos returned NaN because normalized value > 1.0"

    # Compare with eager mode to ensure consistency
    xyz_eager = torch.nn.functional.normalize(c)
    assert torch.allclose(xyz_compiled, xyz_eager, atol=1e-6), "Compiled and eager results differ significantly"

    print("Test passed.")

if __name__ == "__main__":
    test_normalize_compile_cuda_fp32()