import torch
import torch.export
import torch.distributed as dist
import torch.multiprocessing as mp
import warnings
import tempfile
import os

def test_distributed_load_warnings(rank, world_size):
    """
    Test that torch.export.load does not produce duplicate warnings 
    across all ranks in a distributed environment.
    
    This test is derived from Issue #161629, which highlighted that 
    warnings in cpp_extension were printed on all ranks. The fix 
    ensures warnings are rank-0 aware or respect logging levels.
    We apply the same logic to verify torch.export.load behaves correctly.
    """
    # Initialize distributed process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Create a simple module and export it
    class SimpleModule(torch.nn.Module):
        def forward(self, x):
            return x + 1
    
    model = SimpleModule()
    # Use a dummy input for export
    args = (torch.randn(1),)
    ep = torch.export.export(model, args)
    
    # Save the exported program to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt2") as f:
        temp_file = f.name
        torch.export.save(ep, f)

    try:
        # Capture warnings during load
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Load the exported program
            # This is the API under test for distributed warning behavior
            loaded_ep = torch.export.load(temp_file)
            
            # Logic based on Issue #161629 fix:
            # Warnings should generally be restricted to rank 0 to avoid 
            # spamming logs in multi-GPU setups.
            if rank != 0:
                assert len(w) == 0, (
                    f"Rank {rank} unexpectedly emitted {len(w)} warning(s) during load. "
                    f"Warnings: {[str(warning.message) for warning in w]}"
                )
            else:
                # Rank 0 is allowed to emit warnings (e.g., version compatibility, etc.)
                # We just verify the load was successful.
                assert loaded_ep is not None
                
    finally:
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)
        dist.destroy_process_group()

if __name__ == "__main__":
    # Run the test on 2 ranks to simulate a multi-GPU environment
    world_size = 2
    mp.spawn(test_distributed_load_warnings, args=(world_size,), nprocs=world_size, join=True)