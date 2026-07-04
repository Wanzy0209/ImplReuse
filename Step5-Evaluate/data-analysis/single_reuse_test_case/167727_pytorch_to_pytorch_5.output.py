import torch

# Check if MPS is available and if complex64 is supported on MPS
# This prevents the NotImplementedError on environments (like Linux) or builds
# where MPS or complex64 on MPS is not supported.
if not torch.backends.mps.is_available():
    print("MPS device is not available. Skipping test.")
else:
    try:
        # Verify that complex64 allocation works on MPS
        torch.empty(1, dtype=torch.complex64, device="mps")
    except NotImplementedError:
        print("Complex64 is not supported on MPS in this environment. Skipping test.")
    else:
        # Test case for torch.argmin with complex64 on MPS device
        # Based on the original bug report for torch.addmm, checking if similar issues
        # occur with large tensors and complex numbers.

        # success (small tensor)
        a = torch.rand((64, 300), dtype=torch.complex64, device="mps")
        try:
            out_mps = torch.argmin(a, dim=1)
            out_cpu = torch.argmin(a.cpu(), dim=1)
            torch.testing.assert_close(out_mps.cpu(), out_cpu)
            print("Small tensor test passed.")
        except Exception as e:
            print(f"Small tensor test failed: {e}")

        # fails (large tensor) - checking for incorrect results or errors
        a = torch.rand((64, 10000), dtype=torch.complex64, device="mps")
        try:
            out_mps = torch.argmin(a, dim=1)
            out_cpu = torch.argmin(a.cpu(), dim=1)
            torch.testing.assert_close(out_mps.cpu(), out_cpu)
            print("Large tensor test passed.")
        except Exception as e:
            print(f"Large tensor test failed: {e}")