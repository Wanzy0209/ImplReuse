import torch
import torch.nn as nn
from torch.export import export, Dim
from torch.fx import GraphModule
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch._export.utils import _detect_fake_mode_from_gm

class PolarToy(nn.Module):
    def forward(self, x):
        freqs = x.tanh()
        ones_like = torch.ones_like(freqs)
        freqs_cis = torch.polar(ones_like, freqs)
        return freqs_cis.real + freqs_cis.imag

def main():
    mod = PolarToy()
    sample = torch.randn(2, 3, 32)
    dynamic_shapes = ({0: Dim("s0", min=1, max=1024), 1: Dim("s1", min=1, max=2048)},)
    ep = export(mod, (sample,), dynamic_shapes=dynamic_shapes, strict=False)
    gm = ep.module()
    
    print("Before FakeTensorProp:")
    gm.print_readable()
    
    inps = [node.meta.get("val") for node in gm.graph.nodes if node.op == "placeholder"]
    fake_mode = _detect_fake_mode_from_gm(gm)
    FakeTensorProp(gm, fake_mode).propagate(*inps)
    
    print("\nAfter FakeTensorProp:")
    gm.print_readable()

if __name__ == "__main__":
    main()