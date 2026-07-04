import torch

# Ensure reproducibility
torch.manual_seed(0)

# Check for MPS availability to ensure the test runs on supported devices
if torch.backends.mps.is_available():
    # Input tensor (C, H, W) = (4, 6, 7)
    x = torch.randn(4, 6, 7)

    # Adapted model: FractionalMaxPool2d
    # Note: FractionalMaxPool2d requires output_size instead of stride/ceil_mode/divisor_override.
    # We select a valid output_size for the input dimensions (6x7 -> 2x2).
    model = torch.nn.FractionalMaxPool2d(kernel_size=[1, 6], output_size=[2, 2])

    # Run on CPU
    # FractionalMaxPool2d returns (output, indices), so we unpack the result
    out_cpu, _ = model(x)

    # Run on MPS
    out_mps, _ = model(x.to("mps"))

    # Compare results
    if not torch.allclose(out_cpu, out_mps.cpu(), atol=1e-2, rtol=1e-2):
        print("Output does not match!")
        print("CPU Output:")
        print(out_cpu)
        print("MPS Output:")
        print(out_mps)
    else:
        print("Test passed: CPU and MPS outputs match.")
else:
    print("MPS device not available. Skipping test.")