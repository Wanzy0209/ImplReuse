import torch
import functools
import torch.distributed as dist
import tempfile
import os

# Fix for missing torch.compile in older PyTorch versions
if not hasattr(torch, 'compile'):
    print("Warning: torch.compile not found (requires PyTorch 2.0+). Using a pass-through mock.")
    torch.compile = lambda *args, **kwargs: lambda f: f

# Setup for distributed environment (required for new_group)
def setup_distributed():
    if not dist.is_initialized():
        # Using FileStore for single-process testing
        tmpfile = tempfile.NamedTemporaryFile(delete=False)
        tmpfile.close()
        store = dist.FileStore(tmpfile.name, 1)
        dist.init_process_group(
            backend="gloo",
            store=store,
            world_size=1,
            rank=0
        )
        return tmpfile.name
    return None

# Adapted structure from the original bug report
# Original: CustomPolicy -> create_selective_checkpoint_contexts -> partial
# Adapted: GroupConfig -> create_group_options -> partial

class GroupConfig:
    def __init__(self):
        super().__init__()

def create_group_options(config):
    # Returns arguments compatible with torch.distributed.new_group
    return {"ranks": [0], "backend": "gloo"}

# Create a functools.partial similar to the original context_fn1
group_options_fn = functools.partial(create_group_options, GroupConfig())

@torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)
def g():
    # Adaptation: Use the partial to generate arguments for the similar API
    options = group_options_fn()
    # Call the similar API: torch.distributed.new_group
    return dist.new_group(**options)

if __name__ == "__main__":
    tmpfile = None
    try:
        tmpfile = setup_distributed()
        # Execute the compiled function
        group = g()
        print("Test passed: torch.distributed.new_group executed with partial-generated args inside torch.compile")
    except Exception as e:
        print(f"Test failed with error: {e}")
    finally:
        # Cleanup
        if tmpfile and os.path.exists(tmpfile):
            os.remove(tmpfile)
        if dist.is_initialized():
            dist.destroy_process_group()