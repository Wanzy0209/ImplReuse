import torch
torch.manual_seed(0)

model = torch.nn.AvgPool2d(kernel_size=[1, 6], stride=[4, 9], ceil_mode=True, divisor_override=3)
x = torch.randn(4, 6, 7)
out_cpu = model(x)
out_mps = model(x.to("mps"))
if not torch.allclose(out_cpu, out_mps.cpu(), atol=1e-2, rtol=1e-2):
    print("Output does not match!")
    print(out_cpu)
    print(out_mps)