import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18

def test_torch_compile_mps_backward():
    """
    Test case to reproduce the issue where torch.compile fails during loss.backward()
    on the MPS backend.
    """
    # Check if MPS is available to run the test
    if not torch.backends.mps.is_available():
        print("MPS backend is not available. Skipping test.")
        return

    BATCH_SIZE = 4
    NUM_CLASSES = 10
    LEARNING_RATE = 0.01
    device = 'mps'

    # Initialize model, loss, and optimizer
    model = resnet18(num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Move model to MPS device
    model = model.to(device)
    model.train()

    # Apply torch.compile to the training step
    @torch.compile
    def train_step(images, labels):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # The bug occurs specifically here during the backward pass
        loss.backward()
        
        optimizer.step()
        return loss

    # Create dummy data
    images = torch.randn(BATCH_SIZE, 3, 224, 224)
    labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))

    # Execute the compiled function
    try:
        loss = train_step(images, labels)
        # Assertion to ensure the operation completed and returned a valid loss
        assert isinstance(loss, torch.Tensor), "Loss should be a Tensor"
        print("Test Passed: torch.compile with backward() succeeded on MPS.")
    except RuntimeError as e:
        print(f"Test Failed: RuntimeError encountered during backward() on MPS: {e}")
        raise

if __name__ == "__main__":
    test_torch_compile_mps_backward()