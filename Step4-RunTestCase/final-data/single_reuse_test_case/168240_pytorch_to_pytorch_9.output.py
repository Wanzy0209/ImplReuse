import sys
from unittest import mock

# Mock 'requests' and 'urllib3' to prevent import errors related to OpenSSL version incompatibility.
# The test uses torchvision.models with weights=None, so it does not require network requests.
sys.modules['requests'] = mock.MagicMock()
sys.modules['urllib3'] = mock.MagicMock()

import torch
import torchvision
import torch.nn.utils.prune as prune

# Setup model and input similar to the bug report
model = torchvision.models.mobilenet_v2(weights=None)
x = torch.rand((1, 3, 224, 224))

# Select a layer to prune (e.g., the first Conv2d layer in features)
# MobileNetV2 features[0] is a ConvBNReLU block, features[0][0] is Conv2d
module_to_prune = model.features[0][0]

# Apply ln_structured pruning
# Prune 20% of channels with the lowest L2-norm along dimension 0 (output channels)
prune.ln_structured(module_to_prune, name='weight', amount=0.2, n=2, dim=0)

# Verify the pruning mask was created and applied
assert hasattr(module_to_prune, 'weight_mask'), "Pruning mask was not created."
assert torch.any(module_to_prune.weight_mask == 0), "No weights were pruned (mask is all ones)."

# Verify the model can still perform a forward pass
output = model(x)
assert output.shape == (1, 1000), f"Expected output shape (1, 1000), got {output.shape}"