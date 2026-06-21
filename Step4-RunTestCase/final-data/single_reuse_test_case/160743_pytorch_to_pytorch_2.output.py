import torch

torch.manual_seed(0)

# Adapted from AvgPool2d to AvgPool3d
# kernel_size and stride extended to 3 dimensions
model = torch.nn.AvgPool3d(kernel_size=[1, 1, 6], stride=[1, 4, 9], ceil_mode=True, divisor_override=3)

# Input shape adapted from (C, H, W) to (C, D, H, W)
x = torch.randn(4, 5, 6, 7)

out_cpu = model(x)

# Check for MPS availability to avoid RuntimeError on unsupported systems
if torch.backends.mps.is_available():
    out_mps = model(x.to("mps"))

    if not torch.allclose(out_cpu, out_mps.cpu(), atol=1e-2, rtol=1e-2):
        print("Output does not match!")
        print(out_cpu)
        print(out_mps)
    else:
        print("Test passed.")
else:
    print("MPS device not available. Skipping MPS test.")