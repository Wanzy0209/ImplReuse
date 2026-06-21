import torch

torch.manual_seed(0)

# Check if MPS is available to avoid RuntimeError on unsupported platforms
if not torch.backends.mps.is_available():
    print("MPS device is not available. Skipping test.")
else:
    # Adapted to use MultiMarginLoss with specific parameters
    model = torch.nn.MultiMarginLoss(margin=1.0, p=1, reduction='mean')
    # Input: Batch size 4, 6 classes
    x = torch.randn(4, 6)
    # Target: Batch size 4, class indices between 0 and 5
    y = torch.randint(0, 6, (4,))

    out_cpu = model(x, y)
    out_mps = model(x.to("mps"), y.to("mps"))

    if not torch.allclose(out_cpu, out_mps.cpu(), atol=1e-2, rtol=1e-2):
        print("Output does not match!")
        print(out_cpu)
        print(out_mps)
    else:
        print("Test passed.")