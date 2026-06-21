import torch
import torch.distributed as dist
import torch.utils._pytree as pytree
import sys
import os

# Define custom classes from the original bug report
class Foo:
    pass

class Bar:
    def __eq__(self, other):
        return super().__eq__(other)

    def __hash__(self):
        return 0

# Register Bar as a constant in pytree
# Fix: register_constant is not available in torch.utils._pytree.
# We use register_pytree_node to make Bar a leaf node (constant).
def _flatten_bar(obj):
    return [], None

def _unflatten_bar(data, children):
    return Bar()

pytree.register_pytree_node(Bar, _flatten_bar, _unflatten_bar)

# Setup minimal distributed environment to allow torch.distributed.reduce to run
# We use a single process (world_size=1) for a minimal reproducible test
if not dist.is_available():
    print("torch.distributed is not available. Skipping test.")
    sys.exit(0)

# Initialize the process group
dist.init_process_group(
    backend="gloo",
    init_method="tcp://127.0.0.1:29500",
    world_size=1,
    rank=0
)

try:
    # Adapt the original function to include the similar API: torch.distributed.reduce
    @torch.compile(backend="eager")
    def fn(x, obj):
        # Retain the original bug trigger: setting an attribute with a pytree constant
        obj.attr = {3: Bar()}
        
        # Use the similar API: torch.distributed.reduce
        # Since world_size is 1, this is effectively a no-op but valid for the API
        dist.reduce(x, dst=0)
        
        return x + 1

    # Execute the test
    input_tensor = torch.ones(3)
    foo_obj = Foo()
    
    # This call should trigger the compilation and potentially the bug
    # if the guard generation logic fails similarly with torch.distributed.reduce
    result = fn(input_tensor, foo_obj)
    
    # Basic assertion to ensure execution completed
    assert torch.equal(result, torch.ones(3) + 1)
    print("Test passed successfully.")

except Exception as e:
    print(f"Test failed with error: {e}")
    raise
finally:
    # Clean up the distributed environment
    dist.destroy_process_group()