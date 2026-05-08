import torch
input_data = torch.randn(1, 3, 32)
lazy_conv1d = torch.nn.LazyConv1d(out_channels=16, kernel_size=3, stride=1,
    padding=9223372036854775803, bias=True)
lazy_conv1d.to(device=torch.device('cpu'))
output = lazy_conv1d(input_data)