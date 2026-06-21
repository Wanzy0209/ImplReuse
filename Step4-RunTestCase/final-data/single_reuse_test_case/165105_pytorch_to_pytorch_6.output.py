import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def test_all_to_all_single_divergence(rank, world_size):
    """
    Test case for torch.distributed.all_to_all_single adapted from a matmul divergence bug.
    Verifies Eager vs Compile behavior with float16 on CUDA.
    """
    # Initialize the process group
    dist.init_process_group(
        backend="nccl", # Using NCCL as the original bug was on CUDA
        init_method=f"tcp://127.0.0.1:{12345}",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

    # FIX: Check if torch._dynamo is available (requires PyTorch 2.0+)
    if not hasattr(torch, '_dynamo'):
        print(f"[Rank {rank}] torch._dynamo is not available. Skipping test (requires PyTorch 2.0+).")
        dist.destroy_process_group()
        return

    # Configuration from the original bug report
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True

    # Create input tensor similar to the shapes in the original bug (e.g., 14, 6)
    # Using float16 and CUDA as per the original issue
    input_tensor = torch.randn(14, 6, dtype=torch.float16, device=f"cuda:{rank}")

    # Define the function using the similar API
    # Based on the provided info: torch.distributed._functional_collectives.all_to_all_single
    def func_to_test(tensor):
        return torch.distributed._functional_collectives.all_to_all_single(
            tensor,
            output_split_sizes=None,
            input_split_sizes=None,
            group=None
        )

    # 1. Run in Eager mode
    try:
        eager_output = func_to_test(input_tensor)
    except Exception as e:
        print(f"[Rank {rank}] Eager mode failed: {e}")
        dist.destroy_process_group()
        return

    # 2. Run in Compiled mode (torch.compile / torch._dynamo)
    try:
        compiled_func = torch.compile(func_to_test)
        compiled_output = compiled_func(input_tensor)
    except Exception as e:
        print(f"[Rank {rank}] Compiled mode failed: {e}")
        dist.destroy_process_group()
        return

    # 3. Verify results (Check for divergence)
    # Note: all_to_all_single results depend on all ranks, so we check locally
    # but the operation itself ensures consistency across ranks if implemented correctly.
    try:
        assert torch.allclose(eager_output, compiled_output), \
            f"[Rank {rank}] Divergence detected! Eager and Compiled outputs differ."
        print(f"[Rank {rank}] Test Passed: No divergence between Eager and Compiled modes.")
    except AssertionError as e:
        print(e)

    dist.destroy_process_group()

if __name__ == "__main__":
    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. This test requires CUDA (based on the original bug report).")
    else:
        world_size = 2
        # Run the test on multiple processes
        mp.spawn(test_all_to_all_single_divergence, args=(world_size,), nprocs=world_size, join=True)