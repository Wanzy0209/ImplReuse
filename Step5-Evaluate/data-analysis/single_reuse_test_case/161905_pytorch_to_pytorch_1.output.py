import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import os

# Handle the ImportError caused by urllib3/OpenSSL version incompatibility
# by defining a fallback model if torchvision import fails.
try:
    from torchvision.models.resnet import resnet18
except ImportError:
    # Fallback model definition if torchvision is unavailable or has dependency issues
    class resnet18(nn.Module):
        def __init__(self, num_classes=1000):
            super(resnet18, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.fc = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            return self.fc(x)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize the process group
    # Note: 'gloo' is typically used for CPU/Mac, but MPS support in distributed is limited.
    # We use 'gloo' here as the default backend available on macOS.
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    BATCH_SIZE = 4
    NUM_CLASSES = 10
    LEARNING_RATE = 0.01
    device = 'mps'

    # Check if MPS is available, otherwise fallback to CPU to ensure the test runs
    if not torch.backends.mps.is_available():
        if rank == 0:
            print("MPS is not available. Falling back to CPU for this test.")
        device = 'cpu'

    model = resnet18(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    model.train()

    # Create dummy data
    images = torch.randn(BATCH_SIZE, 3, 224, 224)
    labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))

    # Move data to device
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward pass
    loss.backward()

    # Adaptation: Replace torch.compile with torch.distributed.reduce
    # We attempt to reduce the gradients to rank 0 to verify the API behavior on the device.
    try:
        for param in model.parameters():
            if param.grad is not None:
                # dist.reduce(tensor, dst, op=ReduceOp.SUM, ...)
                # This reduces the gradient tensors from all ranks to rank 0.
                dist.reduce(param.grad, dst=0)
        
        if rank == 0:
            print(f"Rank {rank}: torch.distributed.reduce successful on {device} backend.")
    except Exception as e:
        print(f"Rank {rank}: torch.distributed.reduce failed on {device} with error: {e}")

    cleanup()

if __name__ == "__main__":
    # Check if distributed is available
    if not dist.is_available():
        print("torch.distributed is not available. Skipping test.")
    else:
        world_size = 2
        # Spawn processes to simulate distributed training
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)