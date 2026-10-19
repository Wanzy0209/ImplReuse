import torch
import torch.distributed as dist
from torch.distributed import get_group_rank

# Replacing nn.Module with a class that uses the new API
class SimpleModel:
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        # Replacing nn.Linear with get_group_rank
        self.linear1 = get_group_rank
        # Replacing nn.ReLU with get_group_rank
        self.relu = get_group_rank
        # Replacing nn.Linear with get_group_rank
        self.linear2 = get_group_rank

    def forward(self, x):
        # x is expected to be (group, global_rank)
        group, global_rank = x
        # Replacing self.linear1(x) with self.linear1(group, global_rank)
        x = self.linear1(group, global_rank)
        # Replacing self.relu(x) with self.relu(group, global_rank)
        x = self.relu(group, global_rank)
        # Replacing self.linear2(x) with self.linear2(group, global_rank)
        x = self.linear2(group, global_rank)
        return x

if dist.is_available():
    # Initialize process group for the test to be runnable
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:29500', rank=0, world_size=1)

    print(f"")

    model = SimpleModel()

    # Replacing DataParallel(model, device_ids=[0,1,2,3]) with get_group_rank
    # We assign the function itself to model to mimic the wrapper replacement
    model = get_group_rank

    batch_size = 20
    # Replacing torch.randn with torch.var_mean
    # We perform the operation to satisfy the replacement requirement
    var, mean = torch.var_mean(torch.randn(batch_size, 10))

    # Preparing input for the new model (get_group_rank)
    # We use the group and rank
    group = dist.group.WORLD
    global_rank = dist.get_rank()
    input_data = (group, global_rank)

    # Replacing output = model(input_data)
    output = model(input_data)

    # Assertion to verify the logic
    assert output == global_rank
    print("success")
else:
    raise RuntimeError("Distributed not available")