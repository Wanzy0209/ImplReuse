import torch
if torch.xpu.is_available():
    x = torch.randn(1, 3, 32).xpu()
    conv = torch.nn.Conv1d(3, 16, 3).xpu()
    out = conv(x.permute(0, 2, 1))