import torch
from torch.nn import Conv2d

# Adapt input data for Conv2d (Batch, Channel, Height, Width)
input_data = torch.randn(1, 16, 10, 10)


class MyConv2dModule(torch.nn.Module):

    def __init__(self):
        super(MyConv2dModule, self).__init__()
        # Use the same extreme padding value that caused the crash in Conv1d
        self.conv1 = Conv2d(in_channels=16, out_channels=32, kernel_size=3,
            stride=1, padding=9223372036854775803)
        self.add_module(name='conv1', module=self.conv1)


model = MyConv2dModule()
# This call is expected to trigger the error or crash similar to Conv1d
output = model.conv1(input_data)