import torch
import torch.distributed as dist
import sys

# Handle import error for older PyTorch versions where device_mesh might not exist
try:
    from torch.distributed.device_mesh import init_device_mesh, _mesh_resources
except ImportError:
    init_device_mesh = None
    _mesh_resources = None

def test_get_root_mesh_state_isolation():
    """
    Test case for Issue 163330: wrong returns of get_root_mesh in _mesh_resources.
    
    This test verifies that get_root_mesh correctly identifies the parent mesh
    for a sub-mesh, even after other global meshes have been initialized.
    
    The bug manifests when:
    1. mesh1 is created.
    2. mesh2 is created.
    3. get_root_mesh(mesh1_sub) incorrectly returns mesh2 instead of mesh1.
    """
    # Check if the required module is available
    if init_device_mesh is None or _mesh_resources is None:
        print("torch.distributed.device_mesh module not found. Skipping test.")
        return

    if not dist.is_available():
        print("Distributed package not available. Skipping test.")
        return

    if not dist.is_initialized():
        # In a real test environment, this is handled by the test runner (e.g., torchrun)
        # We initialize here to make the script standalone if run with appropriate backend
        try:
            dist.init_process_group("nccl")
        except Exception as e:
            print(f"Failed to initialize process group: {e}")
            sys.exit(1)

    rank = dist.get_rank()
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Step 1: Initialize first mesh (mesh1) ---
    mesh1 = init_device_mesh(
        device_type, (1, 4, 2), mesh_dim_names=("a", "b", "c")
    )
    mesh1_c = mesh1["c"]

    # Verify initial state
    root_1_initial = _mesh_resources.get_root_mesh(mesh1_c)
    if rank == 0:
        print(f"[Rank 0] Initial root of mesh1_c: {root_1_initial.shape}")
        assert root_1_initial is mesh1, "Initial root mesh check failed for mesh1"

    # --- Step 2: Initialize second mesh (mesh2) ---
    # This action triggers the bug in the original implementation by potentially
    # overwriting global state or weak references.
    mesh2 = init_device_mesh(
        device_type, (2, 2, 2), mesh_dim_names=("a", "b", "c")
    )
    mesh2_c = mesh2["c"]

    # Verify mesh2's root
    root_2 = _mesh_resources.get_root_mesh(mesh2_c)
    if rank == 0:
        print(f"[Rank 0] Root of mesh2_c: {root_2.shape}")
        assert root_2 is mesh2, "Root mesh check failed for mesh2"

    # --- Step 3: Verify mesh1's root is still mesh1 (The Bug Check) ---
    # The bug reported in Issue 163330 is that this call returns mesh2.
    root_1_final = _mesh_resources.get_root_mesh(mesh1_c)
    
    if rank == 0:
        print(f"[Rank 0] Final root of mesh1_c: {root_1_final.shape}")
        
        # This assertion will fail if the bug is present.
        # Expected: mesh1 (shape 1, 4, 2)
        # Buggy behavior: mesh2 (shape 2, 2, 2)
        assert root_1_final is mesh1, (
            f"Bug reproduced: get_root_mesh returned the wrong mesh. "
            f"Expected mesh1 (shape {mesh1.shape}), but got mesh with shape {root_1_final.shape}."
        )
        print("Test Passed: get_root_mesh correctly maintains state isolation.")

    dist.destroy_process_group()

if __name__ == "__main__":
    # Usage: torchrun --nproc_per_node=8 test_get_root_mesh.py
    test_get_root_mesh_state_isolation()