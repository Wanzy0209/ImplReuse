import torch
import torch.nn as nn
import torch.optim as optim
import unittest

# Removed: from torchvision.models.resnet import resnet18
# The import caused an ImportError due to an environment incompatibility 
# between urllib3 v2.0 and the OpenSSL version (1.0.2u) in the Python environment.
# To fix this and keep the test logic intact, we define a local minimal ResNet-like model.

class SimpleResNet(nn.Module):
    """
    A minimal ResNet-like model to replace torchvision.models.resnet18.
    This avoids the external dependency on torchvision and the subsequent
    urllib3/OpenSSL import error, while preserving the test logic for 
    torch.compile and MPS backward pass.
    """
    def __init__(self, num_classes=10):
        super(SimpleResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # A simple block to mimic ResNet structure
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        identity = x
        out = self.layer1(x)
        out += identity # Residual connection
        out = self.relu(out)
        
        x = self.avgpool(out)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Translating the pattern of tf.compat.v1.global_variables_initializer
# which checks execution context (eager vs graph) and returns an appropriate Op.
# Here we check the device context and return a compiled training step.

def get_compiled_step_initializer(device):
    """
    Returns a compiled training step function for the specified device.
    This mimics the structure of global_variables_initializer which checks
    context and returns an initialization Op.
    """
    # Check context (MPS availability) similar to context.executing_eagerly()
    if device == 'mps' and not torch.backends.mps.is_available():
        return None # Indicate skip or handle error

    # Define the core operation (similar to variables_initializer logic)
    def _train_step(model, images, labels, optimizer, criterion):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward() # The specific line failing in the bug report
        optimizer.step()
        return loss

    # Apply the compilation (similar to returning the specific Op)
    # The bug is that this fails on MPS during backward()
    return torch.compile(_train_step)

class TestTorchCompileMPS(unittest.TestCase):
    def test_resnet18_backward_on_mps(self):
        device = 'mps'
        
        # Get the compiled function using our initializer pattern
        train_step = get_compiled_step_initializer(device)
        
        if train_step is None:
            self.skipTest("MPS backend is not available.")

        # Setup model and data
        BATCH_SIZE = 4
        NUM_CLASSES = 10
        # Using SimpleResNet instead of resnet18 to avoid import errors
        model = SimpleResNet(num_classes=NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        images = torch.randn(BATCH_SIZE, 3, 224, 224).to(device)
        labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,)).to(device)
        
        # Execute the compiled step
        # This should trigger the bug if not fixed
        try:
            loss = train_step(model, images, labels, optimizer, criterion)
            # If we reach here, the backward pass succeeded
            self.assertIsNotNone(loss)
        except RuntimeError as e:
            self.fail(f"torch.compile failed on MPS backend during backward pass: {e}")

if __name__ == '__main__':
    unittest.main()