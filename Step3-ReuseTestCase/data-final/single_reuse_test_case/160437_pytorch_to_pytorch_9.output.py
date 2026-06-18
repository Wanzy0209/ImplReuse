import torch
import torch.distributed as dist
import os

# Setup distributed environment (required for torch.distributed.broadcast_object_list)
# Using a single-process setup for local testing
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

if not dist.is_initialized():
    dist.init_process_group(backend="gloo", rank=0, world_size=1)

@torch.compile(backend="eager")
def fn(x, i):
    if i == 1:
        # Adaptation: Replace torch._dynamo.graph_break() with the similar API
        # to verify that the compiler handles this specific distributed operation
        # correctly within a conditional branch without generating an empty graph.
        obj_list = [x]
        dist.broadcast_object_list(obj_list, src=0)
        return obj_list[0]
    return x + 1

if __name__ == "__main__":
    inp = torch.randn(3)
    
    # Test case 1: Normal execution path
    out1 = fn(inp, 0)
    assert torch.equal(out1, inp + 1), "Test case 1 failed"

    # Test case 2: Execution path containing the similar API
    # This corresponds to the i=1 case in the original bug report
    out2 = fn(inp, 1)
    assert torch.equal(out2, inp), "Test case 2 failed"

    # Test case 3: Normal execution path again
    out3 = fn(inp, 2)
    assert torch.equal(out3, inp + 1), "Test case 3 failed"

    print("All test cases passed.")