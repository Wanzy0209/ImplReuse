import torch
import torch.nn as nn
import torch.optim as optim
import unittest

# Define a simple model to replace torchvision.models.resnet18
# This avoids the dependency chain that causes the ImportError with urllib3/OpenSSL
class SimpleModel(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Handles variable input sizes like 224x224
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class TestTorchCompileMPSBackward(unittest.TestCase):
    def test_backward_pass_in_compiled_mps_context(self):
        """
        Test that loss.backward() works correctly inside a torch.compile
        function on the MPS backend.
        
        This test leverages the semantic pattern of tf.distribute.get_replica_context:
        verifying the execution context (device/backend) before performing 
        sensitive operations like the backward pass.
        """
        # Setup parameters
        BATCH_SIZE = 4
        NUM_CLASSES = 10
        LEARNING_RATE = 0.01
        device = 'mps'

        # Check for MPS availability
        if not torch.backends.mps.is_available():
            self.skipTest("MPS backend is not available on this system.")

        # Initialize model, loss, and optimizer
        # Replaced resnet18 with SimpleModel to avoid external dependency errors
        model = SimpleModel(num_classes=NUM_CLASSES)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

        # Move model to device and set to training mode
        model = model.to(device)
        model.train()

        # Define the compiled execution context
        @torch.compile
        def train_step(images, labels):
            # Move data to the target device
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Semantic Translation of tf.distribute.get_replica_context:
            # In TensorFlow, one checks the replica context to ensure operations
            # are valid in the current scope. Here, we explicitly verify that
            # the loss tensor resides on the expected 'mps' device context
            # before attempting the backward pass.
            self.assertEqual(loss.device.type, device, 
                             "Loss tensor is not on the expected MPS device context.")

            # Backward pass (The operation failing in the original bug)
            loss.backward()
            
            # Verify gradients were computed
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertIsNotNone(param.grad, 
                                         f"Gradient for {name} was not computed during backward pass.")

            optimizer.step()

        # Create dummy data
        images = torch.randn(BATCH_SIZE, 3, 224, 224)
        labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))

        # Execute the training step
        # If the bug exists, this will raise a RuntimeError during loss.backward()
        try:
            train_step(images, labels)
        except RuntimeError as e:
            self.fail(f"Runtime error occurred during backward pass on MPS: {e}")

if __name__ == '__main__':
    unittest.main()