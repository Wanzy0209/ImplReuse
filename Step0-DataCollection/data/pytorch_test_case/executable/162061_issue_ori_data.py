import torch
import torch.nn as nn
import onnx
from torchsummary import summary


class Model(nn.Module):
    def __init__(self, kernel_size=3, upscale_factor=2):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=kernel_size, padding="same")
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, input):
        x = self.conv(input)
        x = self.pixel_shuffle(x)
        return x


model = Model()
summary(model.cuda(), (1, 128, 128))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

x = torch.ones((1, 1, 128, 128))
x = x.to(device)
torch.onnx.export(
    model,
    x,
    f"dynamo.onnx",
    dynamo=True,
    external_data=False,
    opset_version=20,
    input_names=["input"],
    output_names=["output"],
    dynamic_shapes={"input": {0: "batch", 2: "width", 3: "height"}},
)