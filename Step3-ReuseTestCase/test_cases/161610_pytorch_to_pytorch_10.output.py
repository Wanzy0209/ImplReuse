import torch
import torch.distributed as dist
import os
from typing import NamedTuple

class MyNamedTuple(NamedTuple):
    first: torch.Tensor
    second: torch.Tensor

# We modify the subclass to allow dynamic attributes by including '__dict__' in __slots__.
# This ensures the test case is runnable and demonstrates the persistence issue.
class MyNamedTupleSubclass(MyNamedTuple):
    __slots__ = ('__dict__',)

def main():
    # Initialize distributed environment for single-process testing
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend='gloo', rank=0, world_size=1)

    print("\nTesting NamedTuple with gather_object:")
    extended_tup = MyNamedTupleSubclass(first=torch.tensor([2.0]), second=torch.tensor(1.0))
    
    # Add dynamic attribute
    extra_info = torch.tensor(4.0)
    extended_tup.extra_info = extra_info

    # Use gather_object to transfer the object
    output_list = [None]
    dist.gather_object(extended_tup, object_gather_list=output_list, dst=0)

    # Verify persistence of the dynamic attribute
    gathered_obj = output_list[0]
    try:
        assert gathered_obj.extra_info is not None
        assert torch.equal(gathered_obj.extra_info, extra_info)
        print(f"NamedTuple gather_object result: {gathered_obj.extra_info}")
    except AttributeError:
        print("Error: Dynamic attribute 'extra_info' was not preserved by gather_object")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()