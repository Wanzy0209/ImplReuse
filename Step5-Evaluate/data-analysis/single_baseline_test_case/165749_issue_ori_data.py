# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from torch.optim import SGD

if __name__ == "__main__":
    d = 65
    x = torch.randn((1, 2, 32, 32)).cuda()
    model = torch.compile(weight_norm(nn.Conv2d(2, d, 2)).train().cuda())
    opt = SGD(model.parameters())
    for _ in range(1000):
        model(x).mean().backward()
        opt.step()
        opt.zero_grad()