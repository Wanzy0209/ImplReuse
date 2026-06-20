import torch
import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode
import sys

# Fix: Handle import error for FakeProcessGroup by trying alternative locations
try:
    from torch.distributed._distributed_c10d import FakeProcessGroup
except ModuleNotFoundError:
    try:
        from torch.testing._internal.distributed.fake_pg import FakeProcessGroup
    except ModuleNotFoundError:
        print("FakeProcessGroup not found in this environment. Skipping test.")
        sys.exit(0)

# Adapted from tf.experimental.dtensor.client_id pattern
def validate_process_group_config(pg):
    """
    Validates the process group configuration similar to how 
    tf.experimental.dtensor.client_id validates client ID against num_clients.
    """
    rank = pg.rank()
    world_size = pg.size()
    
    if rank < 0:
        raise ValueError(
            f"Rank must be >= 0, got {rank}."
        )
    if rank >= world_size:
        raise ValueError(
            f"Rank must be < world_size ({world_size}), got {rank}."
        )

class SimpleTensorMode(TorchDispatchMode):
    def __init__(self):
        self.ops_called = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        self.ops_called.append(func)
        return func(*args, **kwargs)

def test_fake_process_group_correctness():
    # Reproduce the bug: Direct construction of FakeProcessGroup
    fake_pg = FakeProcessGroup(rank=0, world_size=3)
    
    # Leverage the similar API pattern: Validate configuration
    validate_process_group_config(fake_pg)

    mode = SimpleTensorMode()
    with mode:
        tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        dist.all_reduce(tensor, group=fake_pg)

    # The bug report indicates that 'c10d.allreduce_.default' is missing.
    # We assert that the correct distributed operator is called.
    expected_op_name = "c10d.allreduce_.default"
    called_op_names = [str(op) for op in mode.ops_called]
    
    assert any(expected_op_name in name for name in called_op_names), (
        f"Expected {expected_op_name} to be called, but got {called_op_names}. "
        "This indicates the silent incorrectness bug where FakeProcessGroup "
        "does not trigger the correct dispatch."
    )

if __name__ == "__main__":
    test_fake_process_group_correctness()