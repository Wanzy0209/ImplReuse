import torch
if torch.xpu.is_available():
    x = torch.randn(1, 3, 10).xpu()
    conv = torch.nn.Conv1d(3, 5, 3).xpu()
    out = conv(x.permute(0, 2, 1))