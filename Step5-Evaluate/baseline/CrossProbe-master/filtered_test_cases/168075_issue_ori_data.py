import torch

class MyModule(torch.nn.Module):
    def forward(self, x):
        print(3**2)
        return x

model = MyModule()
model_script = torch.jit.script(model)
x = torch.rand((10,))
print("Original:")
model(x)
print("Scripted:")
model_script(x)