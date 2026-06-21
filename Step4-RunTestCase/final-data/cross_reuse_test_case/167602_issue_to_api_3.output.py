import torch
import os
from torch.utils.cpp_extension import include_paths

def test_cuda_include_paths_for_ubuntu_environment():
    """
    Test case to verify CUDA include paths are correctly resolved on the system.
    
    This test is derived from Issue #167602 where Stable Diffusion failed on 
    Ubuntu 24.04 due to CUDNN errors. The failure is likely related to how 
    PyTorch detects and configures CUDA library paths, especially regarding 
    Debian/Ubuntu package handling.
    
    The test leverages torch.utils.cpp_extension.include_paths to ensure 
    the environment is correctly configured for CUDA operations.
    """
    # Reproduce the condition: CUDA must be available (as in the bug report)
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        return

    # Leverage the similar API to get the include paths
    # This function contains specific logic for Debian/Ubuntu packages
    # (e.g., checking if cuda_home_include is '/usr/include')
    cuda_include_paths = include_paths(device_type="cuda")

    # Assertion 1: Ensure paths are returned
    assert isinstance(cuda_include_paths, list), "include_paths should return a list"
    assert len(cuda_include_paths) > 0, "No CUDA include paths returned"

    # Assertion 2: Verify that the paths actually exist on the filesystem.
    # If these paths are invalid, CUDNN headers cannot be located, leading to the bug.
    for path in cuda_include_paths:
        assert os.path.exists(path), f"Include path '{path}' does not exist"
        assert os.path.isdir(path), f"Include path '{path}' is not a directory"

    # Assertion 3: Check for the presence of a core CUDA header in the identified paths.
    # This validates that the paths are functional for compilation/runtime linking.
    found_cuda_header = False
    for path in cuda_include_paths:
        if os.path.exists(os.path.join(path, "cuda.h")):
            found_cuda_header = True
            break
    
    assert found_cuda_header, "cuda.h not found in any of the resolved include paths"

if __name__ == "__main__":
    test_cuda_include_paths_for_ubuntu_environment()