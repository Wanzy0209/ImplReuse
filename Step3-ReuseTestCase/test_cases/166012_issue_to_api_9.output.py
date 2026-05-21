import torch
import torch.backends.nnpack as nnpack

def test_nnpack_flags_consistency():
    """
    Test that torch.backends.nnpack.flags correctly manages state consistency,
    ensuring that the global flag is restored to its original value after
    exiting the context manager. This mirrors the concern for consistency
    between different execution states (e.g., cache hit vs miss) raised in
    the original issue.
    """
    # Check if nnpack is available in this build
    if not hasattr(torch.backends, 'nnpack'):
        return

    # Capture the initial state before any context manipulation
    # This represents the "baseline" or "empty cache" state
    initial_state = torch._C._get_nnpack_enabled()

    # Scenario 1: Enable NNPACK inside the context
    # This simulates a "Cache Miss" scenario where state changes
    with nnpack.flags(enabled=True):
        assert torch._C._get_nnpack_enabled() == True, \
            "NNPACK flag was not enabled inside the context manager."

    # Verify restoration (Consistency Check)
    # This simulates a "Cache Hit" or subsequent run where we expect
    # the environment to match the baseline or previous state
    restored_state_1 = torch._C._get_nnpack_enabled()
    assert restored_state_1 == initial_state, \
        f"State inconsistency detected after context exit. Expected {initial_state}, got {restored_state_1}."

    # Scenario 2: Explicitly Disable NNPACK inside the context
    with nnpack.flags(enabled=False):
        assert torch._C._get_nnpack_enabled() == False, \
            "NNPACK flag was not disabled inside the context manager."

    # Verify restoration again
    restored_state_2 = torch._C._get_nnpack_enabled()
    assert restored_state_2 == initial_state, \
        f"State inconsistency detected after second context exit. Expected {initial_state}, got {restored_state_2}."

if __name__ == "__main__":
    test_nnpack_flags_consistency()
    print("Test passed: NNPACK flags state is consistent.")