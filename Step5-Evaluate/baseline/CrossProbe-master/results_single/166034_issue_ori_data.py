# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
from torch.fx.passes.split_module import split_module

class MyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x, y):
        z = self.linear(x + self.param).clamp(min=0.0, max=1.0)
        w = self.linear(y).clamp(min=0.0, max=1.0)
        return z + w

# symbolically trace model
my_module = MyModule()
inputs = (torch.rand(3, 4), torch.rand(3, 4))
my_module_traced = torch.export.export(my_module, inputs)

# random mod partitioning
partition_counter = 0
NPARTITIONS = 3

def mod_partition(node: Node):
    global partition_counter
    partition = partition_counter % NPARTITIONS
    partition_counter = (partition_counter + 1) % NPARTITIONS
    return partition

# split module in module with submodules
module_with_submodules = split_module(
    my_module_traced, my_module, mod_partition
)
print(module_with_submodules)
module_with_submodules(*inputs)