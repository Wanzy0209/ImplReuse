import torch
import torchvision

model = torchvision.models.mobilenet_v2(weights=None)
x = torch.rand((1, 3, 224, 224))
ep = torch.export.export(model, (x,))
torch.testing.assert_close(model(x), ep.module()(x))