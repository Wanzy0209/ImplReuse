import torch
import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode
# Fix: Import FakeProcessGroup from the testing utilities module instead of the private C++ module
from torch.testing._internal.distributed.fake_pg import FakeProcessGroup, FakeStore


class DispatchRecorder(TorchDispatchMode):
    """
    A TorchDispatchMode that records the function names of dispatched ops.
    """
    def __init__(self):
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        self.ops.append(func)
        return func(*args, **kwargs)


def test_fake_process_group_dispatch():
    """
    Test that FakeProcessGroup correctly triggers the dispatch mechanism
    for collective operations like all_reduce.
    
    This test addresses the bug where direct construction of FakeProcessGroup
    resulted in silent incorrectness by not dispatching c10d ops.
    """
    # Setup: Directly construct FakeProcessGroup as in the bug report
    # Fix: The Python FakeProcessGroup uses 'size' instead of 'world_size'
    fake_pg = FakeProcessGroup(rank=0, size=3)
    recorder = DispatchRecorder()

    # Execute: Run all_reduce inside the dispatch mode
    with recorder:
        tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        dist.all_reduce(tensor, group=fake_pg)

    # Verify: Check that the collective operation was actually dispatched
    # The bug report expected 'c10d.allreduce_.default' to be printed/called.
    # We check if any op with 'allreduce' in its name was recorded.
    found_allreduce = False
    for op in recorder.ops:
        if 'allreduce' in op.name.lower():
            found_allreduce = True
            break

    assert found_allreduce, (
        "Expected 'c10d.allreduce_' (or similar) to be dispatched, but it was missing. "
        "This indicates the silent incorrectness bug in FakeProcessGroup is present."
    )


if __name__ == "__main__":
    test_fake_process_group_dispatch()