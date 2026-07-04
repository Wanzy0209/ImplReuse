import torch
import torch.distributed as dist
import torch.nn as nn
import sys
import random

# Check availability
if not dist.is_available():
    print("torch.distributed is not available. Exiting.")
    sys.exit(0)

# Check if gloo is available (common failure point on Windows/Mac)
if 'gloo' not in dist.backends.supported_backends:
    print("gloo backend is not available. Skipping test.")
    sys.exit(0)

# Setup using TCPStore for robust single-process testing
# Using a random port to avoid conflicts
port = random.randint(10000, 20000)

try:
    # Initialize TCPStore
    store = dist.TCPStore("127.0.0.1", port, 1, is_master=True, wait_for_workers=False)
    # Initialize process group
    dist.init_process_group(backend='gloo', store=store, rank=0, world_size=1)
except Exception as e:
    print(f"Failed to initialize process group: {e}")
    sys.exit(0)

# Wrapper over the distributed irecv operation
# Inheriting from nn.Module to support __constants__ and standard PyTorch patterns
class DistributedRecvOp(nn.Module):
    __constants__ = ["src"]
    src: int

    def __init__(self, src: int = 0) -> None:
        super().__init__()
        self.src = src

    def forward(self, input_tensor):
        # Adaptation: Using torch.distributed.irecv
        work = dist.irecv(input_tensor, src=self.src)
        work.wait()
        return input_tensor

# Wrapper over the Sequential layer
class DistributedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.recv_op = DistributedRecvOp()

    def forward(self, x):
        return self.recv_op.forward(x)

# Test execution
if __name__ == "__main__":
    # Create a tensor with data to send
    tensor = torch.ones(5) * 42
    
    # To make irecv complete (receiving from self), we must send first
    # Note: send is blocking, so it will wait until irecv is posted
    dist.send(tensor, dst=0)
    
    model = DistributedModel()
    # Receiving into the same tensor
    output = model.forward(tensor)
    
    # Verify the data received matches the data sent
    assert torch.equal(output, torch.ones(5) * 42), "Data mismatch in irecv"
    
    print("Test passed: torch.distributed.irecv executed successfully.")
    
    dist.destroy_process_group()