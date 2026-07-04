import torch
import torch.distributed as dist
import tempfile
import os

# Replaced NamedTuple with a standard class to support dynamic attributes.
# NamedTuple (tuple subclass) does not support __dict__ or dynamic attributes.
class MyNamedTuple:
    def __init__(self, first: torch.Tensor, second: torch.Tensor):
        self.first = first
        self.second = second

class MyNamedTupleSubclass(MyNamedTuple):
    # Standard classes support __dict__ by default, allowing dynamic attributes
    pass

def main():
    # Setup minimal distributed environment for single-process testing
    tmpfile = tempfile.NamedTemporaryFile(delete=False)
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{tmpfile.name}",
            world_size=1,
            rank=0
        )

        print("\nTesting custom class with broadcast_object_list:")
        extended_tup = MyNamedTupleSubclass(first=torch.tensor([2.0]), second=torch.tensor(1.0))
        
        # Add dynamic attribute
        extra_info = torch.tensor(4.0)
        extended_tup.extra_info = extra_info
        
        # Broadcast the object (simulating the transfer/compile step)
        object_list = [extended_tup]
        dist.broadcast_object_list(object_list, src=0)
        
        # Verify persistence of dynamic attribute
        result = object_list[0]
        print(f"Broadcast result: {result.extra_info}")
        
        # Assertions to verify the behavior
        assert hasattr(result, 'extra_info'), "Dynamic attribute 'extra_info' was lost during broadcast"
        assert torch.equal(result.extra_info, extra_info), "Dynamic attribute value mismatch"

    finally:
        dist.destroy_process_group()
        os.remove(tmpfile.name)

if __name__ == "__main__":
    main()