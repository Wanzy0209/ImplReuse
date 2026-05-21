import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh, _mesh_resources

def test_get_root_mesh_isolation():
    """
    Test that get_root_mesh correctly identifies the root mesh for sub-meshes
    even when multiple meshes are created sequentially.
    
    This test reproduces the bug where creating a second mesh (mesh2) causes
    get_root_mesh to return mesh2 when queried with a sub-mesh of mesh1.
    """
    # Initialize the process group
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    
    rank = dist.get_rank()
    
    # Create the first global mesh (mesh1)
    mesh1 = init_device_mesh(
        "cuda", (1, 4, 2), mesh_dim_names=("a", "b", "c")
    )
    # Get a sub-mesh of mesh1
    mesh1_c = mesh1["c"]
    
    # Verify that the root of mesh1_c is mesh1
    root1 = _mesh_resources.get_root_mesh(mesh1_c)
    if rank == 0:
        print(f"Initial root of mesh1_c: {root1.shape}")
    assert root1 is mesh1, "Root mesh of mesh1_c should be mesh1"
    
    # Create a second global mesh (mesh2)
    mesh2 = init_device_mesh(
        "cuda", (2, 2, 2), mesh_dim_names=("a", "b", "c")
    )
    # Get a sub-mesh of mesh2
    mesh2_c = mesh2["c"]
    
    # Verify that the root of mesh2_c is mesh2
    root2 = _mesh_resources.get_root_mesh(mesh2_c)
    if rank == 0:
        print(f"Root of mesh2_c: {root2.shape}")
    assert root2 is mesh2, "Root mesh of mesh2_c should be mesh2"
    
    # Verify that the root of mesh1_c is STILL mesh1
    # This assertion fails in the bug report (Issue ID: 163330)
    root1_check = _mesh_resources.get_root_mesh(mesh1_c)
    if rank == 0:
        print(f"Root of mesh1_c (after mesh2 creation): {root1_check.shape}")
    
    assert root1_check is mesh1, (
        f"Root mesh of mesh1_c should still be mesh1 after creating mesh2. "
        f"Expected shape {mesh1.shape}, got {root1_check.shape}"
    )
    
    if rank == 0:
        print("Test passed: get_root_mesh correctly isolates mesh resources.")

    dist.destroy_process_group()

if __name__ == "__main__":
    test_get_root_mesh_isolation()