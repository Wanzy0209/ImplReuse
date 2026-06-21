import torch

# Set seed for reproducibility
torch.manual_seed(0)

# Check if MPS is available before attempting to use it
if not torch.backends.mps.is_available():
    print("MPS device is not available. Skipping MPS test.")
else:
    # Adapt the model to use the similar API (MaxPool2d)
    # Note: divisor_override is specific to AvgPool2d and is removed for MaxPool2d
    model = torch.nn.MaxPool2d(kernel_size=[1, 6], stride=[4, 9], ceil_mode=True)

    # Generate input data
    x = torch.randn(4, 6, 7)

    # Compute outputs on CPU and MPS
    out_cpu = model(x)
    out_mps = model(x.to("mps"))

    # Verify consistency between CPU and MPS
    if not torch.allclose(out_cpu, out_mps.cpu(), atol=1e-2, rtol=1e-2):
        print("Output does not match!")
        print("CPU Output:")
        print(out_cpu)
        print("MPS Output:")
        print(out_mps)
    else:
        print("Test passed: CPU and MPS outputs match.")