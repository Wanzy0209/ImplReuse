import torch
torch.set_default_device('cuda') 

inp = torch.randn(8192)
func = torch.exp
out1 = func(inp)
out2 = torch.compile(func)(inp)
out3_high = func(inp.to(torch.float64))
print((out3_high - out1).abs().max())
print((out3_high - out2).abs().max())