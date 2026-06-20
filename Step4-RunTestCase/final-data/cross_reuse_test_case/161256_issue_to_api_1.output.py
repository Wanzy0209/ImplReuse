import torch

# Leveraging the similar API pattern (tf.sysconfig.get_build_info)
# translated to PyTorch semantics to verify the build environment context of the bug.
def get_build_info():
    """
    Mimics tf.sysconfig.get_build_info using PyTorch internals.
    Returns a dictionary describing the build environment.
    """
    return {
        'is_rocm_build': torch.version.hip is not None,
        'rocm_version': torch.version.hip,
        'cuda_version': torch.version.cuda,
        'is_cuda_build': torch.version.cuda is not None
    }

def test_float64_matmul_rocm():
    # Use the translated API to check the environment
    build_info = get_build_info()
    print(f"Build Info: {build_info}")

    # Original bug reproduction logic
    torch.set_default_dtype(torch.float64)

    # The bug is specific to CUDA/ROCm devices
    if not torch.cuda.is_available():
        print("CUDA/ROCm device not available. Skipping GPU test.")
        return

    x = torch.randn(1000, 1000, device="cuda")
    y = torch.randn(1000, 1000, device="cuda")

    # This operation caused a segfault on ROCm 6.4.3
    z = x @ y

    # Assertions to verify the operation completed successfully without crashing
    assert z is not None, "Result is None"
    assert z.shape == (1000, 1000), f"Expected shape (1000, 1000), got {z.shape}"
    assert not torch.isnan(z[0, 0]), "Result contains NaN"
    
    print("Test passed successfully.")

if __name__ == "__main__":
    test_float64_matmul_rocm()