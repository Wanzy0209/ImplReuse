import torch
import torch.nn.functional as F

def test_normalize_precision_with_compile():
    """
    Test case for Issue #163082.
    Verifies that torch.nn.functional.normalize does not produce vectors with norm > 1
    when using torch.compile on CUDA.
    """
    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        return

    # Set seed for reproducibility
    torch.manual_seed(1337)

    # Define the compiled function
    @torch.compile()
    def vec_norm(e_dist):
        return F.normalize(e_dist)

    # Define the eager function for comparison
    def vec_norm_eager(e_dist):
        return F.normalize(e_dist)

    # Setup input tensor on CUDA with float32 dtype
    device = 'cuda'
    # Input vector from the bug report
    c = torch.tensor([[3.799999, 0.0, 0.0]], device=device, dtype=torch.float32)

    print(f"Input vector: {[x.item() for x in c[0]]}")

    # Run with torch.compile
    xyz_compiled = vec_norm(c)
    norm_compiled = torch.norm(xyz_compiled, dim=1).item()
    print(f"Normalized vector (compile): {[x.item() for x in xyz_compiled[0]]}, Norm: {norm_compiled}")

    # Run without torch.compile (eager mode)
    xyz_eager = vec_norm_eager(c)
    norm_eager = torch.norm(xyz_eager, dim=1).item()
    print(f"Normalized vector (eager): {[x.item() for x in xyz_eager[0]]}, Norm: {norm_eager}")

    # Assertion 1: The norm of the normalized vector should be <= 1.0
    # The bug report shows a norm of 1.0000001192092896 which is > 1.0
    assert norm_compiled <= 1.0, \
        f"Failed: Compiled norm {norm_compiled} is greater than 1.0"

    # Assertion 2: The compiled result should be close to the eager result
    # We use a tolerance that is tight enough to catch the specific error (approx 1.19e-07)
    # but loose enough for general floating point noise.
    assert torch.allclose(xyz_compiled, xyz_eager, atol=1e-7), \
        f"Failed: Compiled and eager results differ significantly."

if __name__ == "__main__":
    test_normalize_precision_with_compile()