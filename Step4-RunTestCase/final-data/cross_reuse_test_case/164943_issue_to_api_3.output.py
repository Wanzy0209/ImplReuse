import torch
import sys

# Handle the case where torch.backends.nnpack is not available in the environment
try:
    import torch.backends.nnpack as nnpack
except ModuleNotFoundError:
    print("Skipping test: torch.backends.nnpack is not available in this environment.")
    sys.exit(0)

def test_nnpack_flags_context_manager():
    """
    Test case for torch.backends.nnpack.flags.
    This test verifies that the backend flag can be toggled correctly,
    mirroring the logic of checking/changing backend states as seen in the MPS issue.
    """
    # Get the initial state of the NNPACK backend
    # This mirrors the 'availability check' aspect of the original bug
    initial_state = torch._C._get_nnpack_enabled()

    # Test enabling the backend using the context manager
    with nnpack.flags(True):
        # Assert that the backend is indeed enabled within the context
        assert torch._C._get_nnpack_enabled() is True, "NNPACK should be enabled inside the context"

    # Assert that the state is restored after exiting the context
    assert torch._C._get_nnpack_enabled() == initial_state, "NNPACK state should be restored to original after context"

    # Test disabling the backend using the context manager
    with nnpack.flags(False):
        # Assert that the backend is disabled within the context
        assert torch._C._get_nnpack_enabled() is False, "NNPACK should be disabled inside the context"

    # Assert that the state is restored again
    assert torch._C._get_nnpack_enabled() == initial_state, "NNPACK state should be restored to original after context"

if __name__ == "__main__":
    test_nnpack_flags_context_manager()
    print("Test passed.")