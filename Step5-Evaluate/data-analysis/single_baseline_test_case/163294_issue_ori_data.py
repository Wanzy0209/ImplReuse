# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch

class SetGradCase(torch.nn.Module):
    def forward(self, x):
        with torch.no_grad():
            y = x * 4
        return y

ep = torch.export.export(
    SetGradCase(),
    (torch.randn(6),),
    strict=False,
)
print(ep)

ep2 = torch.export.export(ep.module(), (torch.randn(6),))
print(ep2)