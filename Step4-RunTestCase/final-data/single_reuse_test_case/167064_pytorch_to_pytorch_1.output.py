import torch
import torch.distributed as dist
import torch.distributions as distributions
import os

def test_reduce_does_not_affect_distribution_validation():
    """
    Test that using torch.distributed.reduce (similar API) does not 
    inadvertently change global distribution validation settings, 
    similar to the bug reported in torch.compile context parallel.
    """
    # Check if torch.compile is available (requires PyTorch 2.0+)
    if not hasattr(torch, 'compile'):
        print("Skipping test: torch.compile is not available (requires PyTorch 2.0+)")
        return

    # Setup minimal distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo', rank=0, world_size=1)

    # 1. Explicitly enable distribution validation to detect side effects
    distributions.Distribution.set_default_validate_args(True)

    # 2. Use the similar API: torch.distributed.reduce
    # We wrap it in a function and compile it, as the original bug involved 
    # torch.compile triggering the side effect.
    def reduce_func(tensor):
        dist.reduce(tensor, dst=0)
        return tensor

    # Compile the function to mimic the context of the original bug
    compiled_reduce = torch.compile(reduce_func)
    
    tensor = torch.ones(1)
    compiled_reduce(tensor)

    # 3. Verify that distribution validation is still active.
    # If the bug occurred, set_default_validate_args(False) would have been called,
    # and creating an invalid distribution (e.g., std=0) would not raise an error.
    try:
        # std=0 is invalid, should raise ValueError if validation is True
        distributions.Normal(0, 0)
        # If we reach here, validation was turned off (Bug reproduced)
        assert False, "torch.distributed.reduce (or compile) changed distribution validation args to False!"
    except ValueError:
        # Expected behavior: validation is still True
        pass

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    test_reduce_does_not_affect_distribution_validation()