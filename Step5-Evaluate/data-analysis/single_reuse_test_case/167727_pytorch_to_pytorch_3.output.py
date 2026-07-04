import torch

# Check if MPS is available before running the test
if not torch.backends.mps.is_available():
    print("MPS backend is not available. Skipping test.")
else:
    try:
        # Test case for torch.t adapted from the addmm large tensor bug report.
        # The original bug involved incorrect results for large complex64 tensors on MPS.
        # This test verifies if torch.t behaves correctly under similar conditions.

        # Test with the large dimensions that triggered the issue in addmm (64, 10000)
        a = torch.rand((64, 10000), dtype=torch.complex64, device="mps")

        # Compare MPS result with CPU reference
        torch.testing.assert_close(
            torch.t(a).cpu(), 
            torch.t(a.cpu())
        )

        # Test with the other large dimension involved (10000, 300)
        b = torch.rand((10000, 300), dtype=torch.complex64, device="mps")

        # Compare MPS result with CPU reference
        torch.testing.assert_close(
            torch.t(b).cpu(), 
            torch.t(b.cpu())
        )
    except NotImplementedError as e:
        # Handle cases where specific MPS operators are missing in the build
        print(f"Skipping test: MPS operator not implemented in this environment. Error: {e}")