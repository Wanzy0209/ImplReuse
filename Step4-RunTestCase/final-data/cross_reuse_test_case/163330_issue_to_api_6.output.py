import sys

try:
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh, _mesh_resources
except ImportError as e:
    print(f"Skipping test: {e}")
    print("This test requires PyTorch >= 2.1 to access torch.distributed.device_mesh.")
    sys.exit(0)

def test_get_root_mesh_isolation():
    """
    Test that get_root_mesh correctly identifies the parent mesh
    even after multiple meshes have been initialized.
    
    This test reproduces the issue where creating a second mesh (mesh2)
    causes get_root_mesh to incorrectly return mesh2 when queried with
    a submesh of mesh1 (mesh1_c).
    """
    if not dist.is_initialized():
        dist.init_process_group("nccl")
        
    rank = dist.get_rank()
    
    # Initialize the first global mesh
    mesh1 = init_device_mesh("cuda", (1, 4, 2), mesh_dim_names=("a", "b", "c"))
    mesh1_c = mesh1["c"]
    
    # Verify that the root of mesh1_c is mesh1
    root1 = _mesh_resources.get_root_mesh(mesh1_c)
    assert root1 is mesh1, "Initial root mesh check failed"
    if rank == 0:
        print(f"Check 1 Passed: Root of mesh1_c is mesh1 (shape {root1.shape})")

    # Initialize a second global mesh with the same dimension names but different shape
    mesh2 = init_device_mesh("cuda", (2, 2, 2), mesh_dim_names=("a", "b", "c"))
    mesh2_c = mesh2["c"]
    
    # Verify that the root of mesh1_c is STILL mesh1 (Bug reproduction)
    # The bug reported that this would return mesh2 instead.
    root1_check = _mesh_resources.get_root_mesh(mesh1_c)
    
    if rank == 0:
        print(f"Check 2: Root of mesh1_c is {root1_check.shape}. Expected {mesh1.shape}")
        
    assert root1_check is mesh1, (
        f"Bug reproduced: get_root_mesh(mesh1_c) returned mesh with shape {root1_check.shape} "
        f"instead of mesh1 with shape {mesh1.shape}"
    )
    
    # Verify that the root of mesh2_c is mesh2
    root2 = _mesh_resources.get_root_mesh(mesh2_c)
    assert root2 is mesh2, "Root mesh for mesh2_c check failed"
    
    if rank == 0:
        print("Check 3 Passed: Root of mesh2_c is mesh2")
        print("All tests passed.")

if __name__ == "__main__":
    test_get_root_mesh_isolation()
    dist.destroy_process_group()