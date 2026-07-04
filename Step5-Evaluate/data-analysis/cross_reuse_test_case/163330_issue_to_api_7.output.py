import torch
import torch.distributed as dist

# Handle missing module gracefully for older PyTorch versions
try:
    from torch.distributed.device_mesh import init_device_mesh, _mesh_resources
except (ImportError, ModuleNotFoundError):
    print("Skipping test: torch.distributed.device_mesh is not available. "
          "This module requires PyTorch >= 2.1.")
    import sys
    sys.exit(0)

def test_get_root_mesh_isolation():
    """
    Test that get_root_mesh correctly returns the root mesh for a submesh
    even after other meshes are initialized.
    
    This test reproduces the issue where creating a second global mesh
    causes get_root_mesh to return the wrong mesh for a submesh of the
    first mesh.
    """
    # Initialize process group if not already initialized (e.g., by torchrun)
    if not dist.is_initialized():
        dist.init_process_group("nccl")

    rank = dist.get_rank()

    # 1. Create the first global mesh (mesh1)
    mesh1 = init_device_mesh(
        "cuda", (1, 4, 2), mesh_dim_names=("a", "b", "c")
    )
    
    # 2. Create a submesh of mesh1
    mesh1_c = mesh1["c"]
    
    # 3. Verify that the root of mesh1_c is mesh1
    root_of_mesh1_c = _mesh_resources.get_root_mesh(mesh1_c)
    if rank == 0:
        print(f"Check 1: Root of mesh1_c shape: {root_of_mesh1_c.shape}")
        assert root_of_mesh1_c is mesh1, \
            f"Expected root of mesh1_c to be mesh1, got {root_of_mesh1_c}"
        assert root_of_mesh1_c.shape == (1, 4, 2), \
            f"Expected shape (1, 4, 2), got {root_of_mesh1_c.shape}"

    # 4. Create a second global mesh (mesh2) with a different shape
    mesh2 = init_device_mesh(
        "cuda", (2, 2, 2), mesh_dim_names=("a", "b", "c")
    )
    
    # 5. Create a submesh of mesh2
    mesh2_c = mesh2["c"]
    
    # 6. Verify that the root of mesh1_c is STILL mesh1 (The Bug Check)
    # The bug reported that this step would return mesh2 instead of mesh1
    root_of_mesh1_c_again = _mesh_resources.get_root_mesh(mesh1_c)
    if rank == 0:
        print(f"Check 2: Root of mesh1_c shape after creating mesh2: {root_of_mesh1_c_again.shape}")
        assert root_of_mesh1_c_again is mesh1, \
            f"Expected root of mesh1_c to still be mesh1, got {root_of_mesh1_c_again}. " \
            "This indicates get_root_mesh is returning the wrong mesh."
        assert root_of_mesh1_c_again.shape == (1, 4, 2), \
            f"Expected shape (1, 4, 2), got {root_of_mesh1_c_again.shape}"

    # 7. Verify that the root of mesh2_c is mesh2
    root_of_mesh2_c = _mesh_resources.get_root_mesh(mesh2_c)
    if rank == 0:
        print(f"Check 3: Root of mesh2_c shape: {root_of_mesh2_c.shape}")
        assert root_of_mesh2_c is mesh2, \
            f"Expected root of mesh2_c to be mesh2, got {root_of_mesh2_c}"
        assert root_of_mesh2_c.shape == (2, 2, 2), \
            f"Expected shape (2, 2, 2), got {root_of_mesh2_c.shape}"

    if rank == 0:
        print("All checks passed.")

    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    test_get_root_mesh_isolation()