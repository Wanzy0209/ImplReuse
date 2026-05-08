import torch

def f(xs):
    return xs.split(1, dim=0)

def backend(gm, inps):
    gm.print_readable()

    # class GraphModule(torch.nn.Module):
    #     def forward(self, L_xs_: "f32[2, 2]"):
    #         l_xs_ = L_xs_
            
    #         # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:100 in __torch_function__, code: return func(*args, **kwargs)
    #         split = torch._tensor.split(l_xs_, 1, dim = 0);  l_xs_ = None
    #         getitem: "f32[1, 2]" = split[0]
    #         getitem_1: "f32[1, 2]" = split[1];  split = None
    #         return (getitem, getitem_1)
    return gm

with torch.device("cuda"):
    xs = torch.randn(2, 2, device="cuda")

    # Eager works
    f(xs)

    # This fails with `module 'torch._tensor' has no attribute 'split'`
    torch.compile(f, backend=backend)(xs)

# Outside of device context, this works
# torch.compile(f, backend=backend)(xs)