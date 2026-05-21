import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import DataParallel
import os

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

def test_dataparallel_with_backend_validation():
    # Setup distributed environment to leverage the similar API (get_group_rank)
    # This simulates the validation logic found in the similar API implementation
    if not dist.is_initialized():
        try:
            # Initialize for single-process multi-gpu context to allow API usage
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '29500'
            dist.init_process_group(backend='gloo', rank=0, world_size=1)
        except Exception as e:
            print(f"Distributed init skipped or failed: {e}")

    # Check backend availability (mimicking the user's custom backend check)
    # Using CUDA as a proxy for the custom backend to ensure the test is runnable
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPUs")

        # Leverage similar API: get_group_rank
        # This mirrors the validation logic in the similar API (checking membership/validity)
        # ensuring the process context is valid before running DataParallel
        try:
            group = dist.group.WORLD
            global_rank = dist.get_rank()
            group_rank = dist.get_group_rank(group, global_rank)
            print(f"Process validated with group rank: {group_rank}")
        except ValueError as e:
            print(f"Validation failed (similar to get_group_rank logic): {e}")
            return

        model = SimpleModel()
        model = model.cuda()  # Move to GPU 0

        # Use DataParallel with available devices
        # Preserving the original logic of wrapping the model
        device_ids = [0, 1]
        model = DataParallel(model, device_ids=device_ids)

        batch_size = 20
        input_data = torch.randn(batch_size, 10).cuda()

        output = model(input_data)
        
        # Assertions to verify success
        assert output is not None, "Model output is None"
        assert output.shape[0] == batch_size, f"Output batch size mismatch: {output.shape[0]} != {batch_size}"
        assert output.device.type == 'cuda', "Output is not on CUDA device"
        
        print("Test passed: DataParallel executed successfully with backend validation")
    else:
        print("Test skipped: CUDA backend not available or insufficient devices")

if __name__ == "__main__":
    test_dataparallel_with_backend_validation()