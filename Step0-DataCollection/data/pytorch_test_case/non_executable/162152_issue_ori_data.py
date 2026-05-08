import torch
import torch.nn as nn
from torch.nn import DataParallel

class SimpleModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

if torch.<mybackend>.is_available() and torch.<mybackend>.device_count() > 1:
    print(f"检测到 {torch.<mybackend>.device_count()} 个GPU")

    model = SimpleModel()
    model = model.<mybackend>()  # 移到GPU 0

    model = DataParallel(model, device_ids=[0,1,2,3])

    batch_size = 20
    input_data = torch.randn(batch_size, 10).<mybackend>()

    output = model(input_data)
    print("success")
else:
    raise