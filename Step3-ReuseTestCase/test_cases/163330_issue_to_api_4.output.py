import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh, _mesh_resources
import os

def test_get_root_mesh_consistency():
    """
    Test case to verify that get_root_mesh returns the correct parent mesh
    even after multiple meshes have been initialized.
    
    Bug Description:
    The _mesh_resources.get_root_mesh returns the wrong mesh (mesh2) when queried
    for the root of a submesh (mesh1_c) belonging to a previously initialized mesh (mesh1),
    if a new mesh (mesh2) was initialized in between.
    """
    # Initialize the process group
    dist.init_process_group("nccl")
    
    rank = dist.get_rank()
    if rank == 0:
        print("Starting test for get_root_mesh consistency...")

    # 1. Initialize the first global mesh (mesh1)
    mesh1 = init_device_mesh(
        "cuda", (1, 4, 2), mesh_dim_names=("a", "b", "c")
    )
    mesh1_c = mesh1["c"]
    
    # 2. Verify that the root of mesh1_c is mesh1
    root_of_mesh1_c = _mesh_resources.get_root_mesh(mesh1_c)
    assert root_of_mesh1_c is mesh1, \
        f"Rank {rank}: Initial check failed. Expected mesh1, got {root_of_mesh1_c}"
    
    if rank == 0:
        print(f"Step 1 OK: Root of mesh1_c is mesh1 (shape {mesh1.shape})")

    # 3. Initialize a second global mesh (mesh2)
    mesh2 = init_device_mesh(
        "cuda", (2, 2, 2), mesh_dim_names=("a", "b", "c")
    )
    mesh2_c = mesh2["c"]
    
    # 4. Verify that the root of mesh2_c is mesh2
    root_of_mesh2_c = _mesh_resources.get_root_mesh(mesh2_c)
    assert root_of_mesh2_c is mesh2, \
        f"Rank {rank}: Check for mesh2 failed. Expected mesh2, got {root_of_mesh2_c}"
        
    if rank == 0:
        print(f"Step 2 OK: Root of mesh2_c is mesh2 (shape {mesh2.shape})")

    # 5. CRITICAL CHECK: Verify that the root of mesh1_c is STILL mesh1
    # This is where the bug manifests (it returns mesh2 instead)
    root_of_mesh1_c_check = _mesh_resources.get_root_mesh(mesh1_c)
    
    if rank == 0:
        print(f"Step 3: Checking root of mesh1_c again...")
        print(f"  Expected shape: {mesh1.shape}")
        print(f"  Got shape:      {root_of_mesh1_c_check.shape}")

    assert root_of_mesh1_c_check is mesh1, \
        f"Rank {rank}: BUG REPRODUCED! Expected root of mesh1_c to be mesh1, but got {root_of_mesh1_c_check}"

    if rank == 0:
        print("Test PASSED: get_root_mesh correctly identifies the parent mesh.")

    dist.destroy_process_group()

if __name__ == "__main__":
    # Check for distributed environment variables
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        test_get_root_mesh_consistency()
    else:
        print("Error: This test must be launched with torchrun.")
        print("Usage: torchrun --nproc_per_node=8 <script_name>")