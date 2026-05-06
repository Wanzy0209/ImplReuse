import torch

class Neuron(torch.nn.Module):
    def __init__(self, n_dims: int = 5, n_targets: int = 3):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, x, y):
        return torch.sigmoid(self.linear(x + y))

class Wrapped(Neuron):
    def forward(self, *args):
        return super().forward(*args)

args = (torch.randn(2, 5), torch.randn(2, 5))
batch = torch.export.Dim.DYNAMIC

compiled = torch.export.export(
    Neuron(), args, dynamic_shapes=({0: batch}, {0: batch})
)
expected = Neuron()(*args)
mod = compiled.module()
got = mod(*args)

compiled = torch.export.export(
    Wrapped(), args, dynamic_shapes=({"args": ({0: batch}, {0: batch})})
)
expected = Wrapped()(*args)
mod = compiled.module()
got = mod(*args)  # fails here: NameError: name 'L' is not defined