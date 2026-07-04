# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
from torch.nn import Conv1d
input_data = torch.randn(1, 16, 100)


class MyConv1dModule(torch.nn.Module):

    def __init__(self):
        super(MyConv1dModule, self).__init__()
        self.conv1 = Conv1d(in_channels=16, out_channels=32, kernel_size=3,
            stride=1, padding=9223372036854775803)
        self.add_module(name='conv1', module=self.conv1)


model = MyConv1dModule()
output = model.conv1(input_data)