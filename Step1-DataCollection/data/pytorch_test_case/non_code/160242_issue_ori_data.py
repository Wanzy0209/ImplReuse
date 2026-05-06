import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()

    def initialize_components(self):
        raise NotImplementedError("Subclasses must implement this method")

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement this method")

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_output_shape(self, input_shape):
        raise NotImplementedError("Subclasses must implement this method")

class MyModel(BaseModule):
    def __init__(self, in_channels=3, out_channels_list=[16, 32, 64], fc_out_channels_list=[128, 256, 2]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels_list = out_channels_list
        self.fc_out_channels_list = fc_out_channels_list
        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.process_function = F.relu  # Default activation after conv layers

        self.initialize_components()
        self.init_weights()

    def initialize_components(self):
        # Initialize convolutional layers
        for out_channels in self.out_channels_list:
            self.convs.append(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=3, padding=1)
            )
            self.in_channels = out_channels

        # Initialize "fully connected" layers (using Conv2d as per original code)
        for i, out_channels in enumerate(self.fc_out_channels_list):
            in_channels = (
                self.out_channels_list[-1]
                if i == 0
                else self.fc_out_channels_list[i - 1]
            )
            self.fcs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )

    def forward(self, x):
        for conv in self.convs:
            x = cp.checkpoint(conv, x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)

        # Apply process function (e.g., activation)
        x = self.process_function(x)

        for fc in self.fcs[:-1]:
            x = cp.checkpoint(fc, x)
            x = F.relu(x)

        x = self.fcs[-1](x)
        return x

    def get_output_shape(self, input_shape):
        H, W = input_shape[2:]
        for _ in range(len(self.out_channels_list)):
            H, W = H // 2, W // 2
        return (
            input_shape[0],
            self.fc_out_channels_list[-1],
            H,
            W,
        )

def my_model_function():
    return MyModel()

def GetInput():
    input_shape = (1, 3, 32, 32)
    return torch.randn(input_shape).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

def set_seed(seed=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_with_torch_compile():
    model = MyModel().to("cuda")
    inp64 = GetInput().to(dtype=torch.float64)
    model.eval()

    with torch.no_grad():
        eager_out = model.to(dtype=torch.float64)(inp64)

    compiled_model = torch.compile(model.to(dtype=torch.float32), mode="max-autotune")
    inp32 = inp64.clone().to(dtype=torch.float32)
    with torch.no_grad():
        compiled_out = compiled_model(inp32)

    torch.testing.assert_close(
        inp32, inp64,
        rtol=1e-2, atol=1e-3, equal_nan=False,
            check_device=False,
            check_dtype=False,
            check_layout=True,
            check_stride=True,
            msg=None
    )
    torch.testing.assert_close(
        eager_out, compiled_out,
        rtol=1e-2, atol=1e-3, equal_nan=False,
            check_device=False,
            check_dtype=False,
            check_layout=True,
            check_stride=True,
            msg=None
    )
    print("Output is ok!")

if __name__ == "__main__":
    set_seed(4)  # Repro mismatch
    check_with_torch_compile()