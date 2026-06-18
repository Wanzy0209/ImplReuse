import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh, _mesh_resources

def test_get_root_mesh_isolation():
    """
    Test case to verify that get_root_mesh correctly returns the parent mesh
    for a sub-mesh, even when multiple global meshes are initialized.
    
    This reproduces the bug where get_root_mesh(mesh1_sub) returns mesh2
    if mesh2 was initialized after mesh1.
    """
    # Initialize the process group
    dist.init_process_group("nccl")

    # Create the first global mesh (mesh1)
    mesh1 = init_device_mesh(
        "cuda", (1, 4, 2), mesh_dim_names=("a", "b", "c")
    )
    # Get a sub-mesh from mesh1
    mesh1_c = mesh1["c"]

    # Verify initial state: root of mesh1_c should be mesh1
    # (This check is implicit in the bug report, but good for completeness)
    assert _mesh_resources.get_root_mesh(mesh1_c) is mesh1, \
        "Initial root check failed: mesh1_c root should be mesh1"

    # Create the second global mesh (mesh2)
    mesh2 = init_device_mesh(
        "cuda", (2, 2, 2), mesh_dim_names=("a", "b", "c")
    )
    # Get a sub-mesh from mesh2
    mesh2_c = mesh2["c"]

    # Verify that mesh2_c correctly identifies mesh2 as its root
    assert _mesh_resources.get_root_mesh(mesh2_c) is mesh2, \
        "Root check failed: mesh2_c root should be mesh2"

    # BUG REPRODUCTION CHECK:
    # After initializing mesh2, calling get_root_mesh on mesh1_c should still
    # return mesh1, not mesh2.
    root_of_mesh1_c = _mesh_resources.get_root_mesh(mesh1_c)
    
    assert root_of_mesh1_c is mesh1, \
        f"Bug reproduced: Expected root of mesh1_c to be mesh1 (shape {mesh1.shape}), " \
        f"but got {root_of_mesh1_c.shape} (mesh2)."

    if dist.get_rank() == 0:
        print("Test passed: get_root_mesh correctly isolates root meshes.")

    dist.destroy_process_group()

if __name__ == "__main__":
    test_get_root_mesh_isolation()