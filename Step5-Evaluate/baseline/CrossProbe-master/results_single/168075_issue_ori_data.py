# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch


class MyModule(torch.nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(3**2)
        return x


model = MyModule()
model_script = torch.jit.script(model)
x = torch.rand((10,))
print("Original model:")
model(x)
print("Scripted model:")
model_script(x)