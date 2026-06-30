import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import sys

# Leveraging the pattern from tf.experimental.enable_strict_mode
# which sets a global state to control behavior.
# Here we define a global flag and a function to enable the GLOO environment configuration.
GLOO_ENV_CONFIGURED = False

def enable_gloo_environment():
    """
    Configures the global environment for the GLOO backend.
    This mimics the pattern of tf.experimental.enable_strict_mode by setting
    a global state (environment variables and a flag) before execution.
    """
    global GLOO_ENV_CONFIGURED
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    GLOO_ENV_CONFIGURED = True

def run_test(rank, world_size):
    """
    Worker function that attempts to run dist.all_to_all with the GLOO backend.
    Based on Issue #162248, this should raise a RuntimeError.
    """
    # Initialize process group with 'gloo' backend
    dist.init_process_group(
        backend='gloo',
        rank=rank,
        world_size=world_size
    )

    # Prepare input and output tensors as per the original issue reproduction
    # Input: list of tensors derived from rank
    input_tensor_list = list((torch.arange(world_size, dtype=torch.int64) + rank * world_size).chunk(world_size))
    # Output: list of empty tensors to receive data
    output_tensor_list = list(torch.empty([world_size], dtype=torch.int64).chunk(world_size))

    try:
        # Attempt the operation that is documented but not supported in GLOO
        dist.all_to_all(output_tensor_list, input_tensor_list)
        
        # If we reach here, the behavior has changed (bug might be fixed or docs were right)
        print(f"Rank {rank}: UNEXPECTED - dist.all_to_all succeeded. Documentation might be correct now.")
        sys.exit(1)
        
    except RuntimeError as e:
        error_msg = str(e)
        # Updated the expected error message to match the actual PyTorch output
        if "ProcessGroup gloo does not support alltoall" in error_msg:
            print(f"Rank {rank}: SUCCESS - Caught expected RuntimeError: {error_msg}")
        else:
            print(f"Rank {rank}: FAILURE - Caught RuntimeError but with unexpected message: {error_msg}")
            sys.exit(1)
    finally:
        dist.destroy_process_group()

def main():
    # Enable the environment configuration (mimicking the similar API's pattern)
    enable_gloo_environment()
    
    world_size = 2
    
    # Use torch.multiprocessing.spawn to run distributed test
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()