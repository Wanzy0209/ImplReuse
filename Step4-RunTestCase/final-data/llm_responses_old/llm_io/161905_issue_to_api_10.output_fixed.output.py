import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Removed torchvision import to avoid the urllib3/OpenSSL environment error.
# Defined a simple CNN model to replace resnet18 for the purpose of testing
# torch.compile on the MPS backend.

class SimpleCNN(nn.Module):
    """
    A simple CNN model to replace torchvision.models.resnet18.
    This avoids the urllib3 import error while maintaining the logic
    of testing a CNN model with torch.compile on MPS.
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class TestMPSCompileBackward(unittest.TestCase):
    """
    Test case to reproduce the issue where torch.compile fails during 
    loss.backward() on the MPS backend.
    """

    @unittest.skipIf(not torch.backends.mps.is_available(), "MPS backend not available")
    def test_resnet18_mps_compile_backward(self):
        BATCH_SIZE = 4
        NUM_CLASSES = 10
        LEARNING_RATE = 0.01
        device = 'mps'

        # Initialize model, loss, and optimizer
        # Using SimpleCNN instead of resnet18 to avoid environment dependency issues
        model = SimpleCNN(num_classes=NUM_CLASSES)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

        # Move model to device
        model = model.to(device)
        model.train()

        # Leverage pattern from similar API (TFRecordDatasetV2 example): 
        # Use numpy for random data generation before converting to tensors
        images_np = np.random.randn(BATCH_SIZE, 3, 224, 224).astype(np.float32)
        labels_np = np.random.randint(0, NUM_CLASSES, (BATCH_SIZE,))

        images = torch.from_numpy(images_np).to(device)
        labels = torch.from_numpy(labels_np).to(device)

        # Define the compiled training step (Original API Under Test)
        @torch.compile
        def train_step(imgs, lbls):
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            # The bug occurs specifically here on MPS
            loss.backward()
            optimizer.step()
            return loss

        # Execute the training step
        try:
            loss = train_step(images, labels)
        except RuntimeError as e:
            self.fail(f"torch.compile failed on MPS backend during backward pass: {e}")

        # Assertions to verify correct execution
        self.assertIsNotNone(loss)
        self.assertFalse(torch.isnan(loss), "Loss is NaN")

        # Verify that gradients were computed (backward pass succeeded)
        grad_found = False
        for param in model.parameters():
            if param.grad is not None:
                self.assertFalse(torch.isnan(param.grad).any(), "Gradients contain NaN")
                grad_found = True
                break
        
        self.assertTrue(grad_found, "No gradients were computed")

if __name__ == '__main__':
    unittest.main()