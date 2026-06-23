import sys
import torch
import torch.nn as nn
from unittest.mock import MagicMock

# Mock torchvision to avoid the urllib3/OpenSSL import error
sys.modules['torchvision'] = MagicMock()
sys.modules['torchvision.models'] = MagicMock()

# Define a simple model to replace torchvision.models.mobilenet_v2
# This ensures the test logic (DataParallel) can run without the broken dependency.
class SimpleMobileNetV2(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 1000)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# Inject the mock model into the mocked torchvision module
import torchvision.models
torchvision.models.mobilenet_v2 = SimpleMobileNetV2

# DataParallel is primarily used for multi-GPU training, so we check for CUDA availability.
if torch.cuda.is_available():
    # This line now uses the mocked model
    model = torchvision.models.mobilenet_v2(weights=None).cuda()
    x = torch.rand((1, 3, 224, 224)).cuda()

    # Wrap the model with the similar API: torch.nn.DataParallel
    dp_model = torch.nn.DataParallel(model)

    # Verify that the DataParallel model produces the same output as the original model
    torch.testing.assert_close(model(x), dp_model(x))
else:
    print("Skipping test: CUDA is not available for torch.nn.DataParallel.")