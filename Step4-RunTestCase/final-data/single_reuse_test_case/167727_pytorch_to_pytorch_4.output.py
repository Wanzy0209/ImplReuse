import torch

# MPS is only available on macOS. The test is running on a Linux environment (/home/ubuntu),
# so we need to check for availability to avoid NotImplementedError.
if torch.backends.mps.is_available():
    # torch.argmax does not support complex64, so we use float32 to test large tensor behavior on MPS
    # success
    a = torch.rand((64, 300), dtype=torch.float32, device="mps")
    out_mps = torch.argmax(a, dim=1)
    out_cpu = torch.argmax(a.cpu(), dim=1)
    torch.testing.assert_close(out_mps.cpu(), out_cpu)

    # large tensor test
    a = torch.rand((64, 10000), dtype=torch.float32, device="mps")
    out_mps = torch.argmax(a, dim=1)
    out_cpu = torch.argmax(a.cpu(), dim=1)
    torch.testing.assert_close(out_mps.cpu(), out_cpu)
else:
    print("Skipping test: MPS backend is not available on this system.")