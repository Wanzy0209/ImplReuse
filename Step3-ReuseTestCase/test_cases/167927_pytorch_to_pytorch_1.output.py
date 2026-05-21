import torch
import torch.distributed as dist
import sys
import os

def setup_distributed():
    """
    Initializes a single-process distributed environment for testing purposes.
    This allows torch.distributed.reduce to be called without a multi-process setup.
    """
    if not dist.is_available():
        print("torch.distributed is not available. Skipping test.")
        sys.exit(0)

    if not dist.is_initialized():
        # Use gloo backend for CPU testing
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29500",
            rank=0,
            world_size=1
        )

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def test_distributed_reduce_with_fullgraph():
    """
    Test case to verify if torch.distributed.reduce works inside torch.compiler.disable
    when using torch.compile with fullgraph=True.
    
    This is adapted from the bug report regarding torch.compile(fullgraph=True) 
    and torch.compiler.disable, replacing the generic function call with 
    torch.distributed.reduce.
    """
    setup_distributed()

    def func_to_compile(x):
        y = x + 1
        
        # The bug report highlights that torch.compiler.disable should be accepted
        # as an intentional graph break even with fullgraph=True.
        with torch.compiler.disable():
            # We use torch.distributed.reduce here as the similar API to test.
            # This operation is often a candidate for exclusion from compilation.
            # dist.reduce modifies y in-place.
            dist.reduce(y, dst=0)
            
        return y + 2

    try:
        # Compile with fullgraph=True
        # According to the bug, this currently raises:
        # torch._dynamo.exc.Unsupported: Skip calling `torch.compiler.disable()`d function
        compiled_func = torch.compile(func_to_compile, fullgraph=True)
        
        input_tensor = torch.ones(4, 4)
        output = compiled_func(input_tensor)
        
        # Verify the result
        # y = x + 1 -> reduce (no-op for single rank) -> y + 2
        # Expected: x + 3
        expected = input_tensor + 3
        assert torch.equal(output, expected), f"Output mismatch. Expected {expected}, got {output}"
        
        print("Test Passed: torch.distributed.reduce works inside torch.compiler.disable with fullgraph=True")

    except torch._dynamo.exc.Unsupported as e:
        print(f"Test Failed (Bug Reproduced): {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Test Failed with unexpected error: {e}")
        sys.exit(1)
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    test_distributed_reduce_with_fullgraph()