import torch
import pytest

def test_distributed_is_available_consistency():
    """
    Test that torch.distributed.is_available() correctly reflects the
    build configuration and internal state.
    
    This mirrors the logic in the bug report where torch.cuda.get_arch_list()
    is checked to verify if specific capabilities (sm_52) are present in the
    current installation. Here we verify that the availability check for
    the distributed package is consistent with the underlying C++ extension.
    """
    # Check the public API
    is_available = torch.distributed.is_available()

    # Verify the return type
    assert isinstance(is_available, bool), \
        f"torch.distributed.is_available() should return bool, got {type(is_available)}"

    # Verify consistency with the internal implementation detail
    # (This mirrors checking the arch list against the actual binaries)
    has_internal_init = hasattr(torch._C, "_c10d_init")
    assert is_available == has_internal_init, \
        f"Mismatch: is_available()={is_available} but hasattr(torch._C, '_c10d_init')={has_internal_init}"

    # If available, ensure the package is actually functional/accessible
    if is_available:
        # This mimics the user checking if their specific arch (sm_52) is in the list.
        # Here we check if the distributed module is actually usable.
        import torch.distributed
        # Check for a known function to ensure the module is loaded correctly
        assert hasattr(torch.distributed, 'is_available')