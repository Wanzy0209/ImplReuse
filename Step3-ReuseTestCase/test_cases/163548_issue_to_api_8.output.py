import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard
import time

def test_solve_on_distributed_tensors():
    """
    Test case leveraging torch.linalg.solve (similar API to torch.solve) 
    on the distributed tensor setup from the original issue.
    
    This test preserves the logic of creating many sharded tensors and 
    measures the execution time of the linear algebra operation, 
    contrasting with the slow planner validation in the original bug.
    """
    # Setup distributed environment
    pg = torch.distributed.init_process_group(backend="gloo")
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)
    device_mesh = init_device_mesh("cpu", (world_size,))

    # Create data similar to the issue, but adapted for torch.linalg.solve
    # We need square matrices A and B.
    # Using 1024 tensors to match the issue's scale of "many tensors"
    num_tensors = 1024
    dim = 128  # Reduced dimension for test execution speed, but count remains high

    # Create A (Identity) and B (Ones)
    # Using eye ensures the matrix is non-singular for solving
    list_A = [torch.eye(dim) for _ in range(num_tensors)]
    list_B = [torch.ones(dim, 1) for _ in range(num_tensors)]

    # Shard the tensors
    sharded_A = [distribute_tensor(t, device_mesh, [Shard(0)]) for t in list_A]
    sharded_B = [distribute_tensor(t, device_mesh, [Shard(0)]) for t in list_B]

    # Measure time for the similar API (torch.linalg.solve)
    # This replaces the slow planner validation from the original issue
    if rank == 0:
        start = time.time()

    # Perform torch.linalg.solve on the sharded tensors
    # Note: torch.solve is deprecated, so we use torch.linalg.solve
    results = []
    for a, b in zip(sharded_A, sharded_B):
        # Attempting to solve. 
        # Note: Depending on PyTorch version, torch.linalg.solve might require 
        # local tensors or specific DTensor support. This tests the API 
        # compatibility with the distributed setup.
        res = torch.linalg.solve(a, b)
        results.append(res)

    if rank == 0:
        end = time.time()
        print(f"Solve operation cost {end - start}s")
        # Basic assertion to ensure execution
        assert len(results) == num_tensors

    dist.barrier()

if __name__ == "__main__":
    test_solve_on_distributed_tensors()