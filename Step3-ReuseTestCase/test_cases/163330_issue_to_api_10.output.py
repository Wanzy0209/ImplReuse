import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh, _mesh_resources

def test_get_root_mesh_reference_integrity():
    """
    Test to verify that get_root_mesh maintains correct references to parent meshes
    even after new meshes are initialized.
    
    This reproduces the issue where creating a second mesh causes get_root_mesh
    to return the wrong parent for a submesh of the first mesh.
    """
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    
    # Initialize the first mesh
    mesh1 = init_device_mesh("cuda", (1, 4, 2), mesh_dim_names=("a", "b", "c"))
    mesh1_c = mesh1["c"]
    
    # Verify initial root reference
    root1 = _mesh_resources.get_root_mesh(mesh1_c)
    assert root1 is mesh1, f"Rank {rank}: Initial root check failed. Expected mesh1, got {root1.shape}"
    
    if rank == 0:
        print(f"Rank 0: Initial root of mesh1_c is mesh1 (shape {mesh1.shape})")

    # Initialize a second mesh with the same dimension names
    mesh2 = init_device_mesh("cuda", (2, 2, 2), mesh_dim_names=("a", "b", "c"))
    mesh2_c = mesh2["c"]
    
    # Verify root reference for the second mesh
    root2 = _mesh_resources.get_root_mesh(mesh2_c)
    assert root2 is mesh2, f"Rank {rank}: Second mesh root check failed. Expected mesh2, got {root2.shape}"
    
    if rank == 0:
        print(f"Rank 0: Root of mesh2_c is mesh2 (shape {mesh2.shape})")

    # Re-verify root reference for the first mesh's submesh
    # This is the critical check that fails in the reported bug
    root1_check = _mesh_resources.get_root_mesh(mesh1_c)
    
    if rank == 0:
        print(f"Rank 0: Re-checking root of mesh1_c after mesh2 creation...")
        
    assert root1_check is mesh1, \
        f"Rank {rank}: BUG REPRODUCED! Expected root of mesh1_c to be mesh1, but got {root1_check.shape}"

    if rank == 0:
        print("Rank 0: Test passed. Root references are correct.")

    dist.destroy_process_group()

if __name__ == "__main__":
    test_get_root_mesh_reference_integrity()