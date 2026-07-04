import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import os

# Removed torchvision import to avoid urllib3/OpenSSL version conflicts in the environment

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Use gloo backend as it is compatible with CPU/MPS coordination
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# Define a simple model to replace torchvision.models.resnet18
# This avoids the dependency on torchvision which triggers the urllib3 import error
class SimpleModel(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def run_test(rank, world_size):
    setup(rank, world_size)
    
    # Check for MPS availability, fallback to CPU if not
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    BATCH_SIZE = 4
    NUM_CLASSES = 10
    
    # Model setup
    # Replaced resnet18 with SimpleModel to avoid import errors
    model = SimpleModel(num_classes=NUM_CLASSES)
    model = model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Adapted train function
    # Note: We remove @torch.compile because torch.compile does not generally support
    # distributed collectives like gather_object inside the compiled graph.
    # The goal is to test the similar API (gather_object) on the MPS backend.
    def train(images, labels, rank, world_size):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Original call site: loss.backward()
        # Adapted call site: torch.distributed.gather_object
        # We gather the loss value (as a float) to rank 0
        
        gather_list = [None] * world_size if rank == 0 else None
        dist.gather_object(loss.item(), gather_list, dst=0)
        
        if rank == 0:
            print(f"Rank 0 gathered losses: {gather_list}")
            # Verify that we gathered the correct number of items
            assert len(gather_list) == world_size
            # Verify that the items are floats (losses)
            assert all(isinstance(x, float) for x in gather_list)
        
        # We still perform backward to complete the training step logic
        loss.backward()
        optimizer.step()

    images = torch.randn(BATCH_SIZE, 3, 224, 224)
    labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
    
    train(images, labels, rank, world_size)
    
    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Use spawn to launch processes
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)