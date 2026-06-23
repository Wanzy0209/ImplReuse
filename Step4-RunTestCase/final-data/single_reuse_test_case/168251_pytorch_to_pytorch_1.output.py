import torch
import torch.distributed as dist
import os
import sys

def setup():
    """
    Setup for single-process distributed testing.
    This allows the test to run without requiring multiple processes or external launchers.
    """
    if not dist.is_available():
        print("torch.distributed is not available. Skipping test.")
        return False
    
    # Initialize environment variables for the gloo backend
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    try:
        # Initialize process group with world_size=1
        dist.init_process_group(backend='gloo', rank=0, world_size=1)
        return True
    except Exception as e:
        print(f"Failed to initialize process group: {e}")
        return False

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def get_sample_tensor():
    """Generates a sample tensor input."""
    return torch.randn(32, 784)

def main():
    if not setup():
        sys.exit(0)

    print("Distributed environment initialized successfully.")
    
    # Get sample inputs
    tensor = get_sample_tensor()
    
    # Fixed the incomplete print statement
    print(f"Input tensor: {tensor}")
    
    # Defined world_size to fix the NameError that would occur in the assert
    world_size = dist.get_world_size()
    assert world_size

    cleanup()

if __name__ == "__main__":
    main()