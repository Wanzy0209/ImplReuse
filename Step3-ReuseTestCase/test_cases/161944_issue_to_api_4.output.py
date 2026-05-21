import torch
from torch.backends.mha import get_fastpath_enabled, set_fastpath_enabled

# Preserve the original bug reproduction logic: 
# Comparing the result of eager execution vs torch.compile.
# Adapted to the similar API: torch.backends.mha.get_fastpath_enabled.

# Save original state to ensure the test is non-destructive
original_state = get_fastpath_enabled()

try:
    # Toggle the state to test dynamic behavior
    set_fastpath_enabled(not original_state)

    def check_fastpath_status():
        return get_fastpath_enabled()

    # Eager execution
    out1 = check_fastpath_status()

    # Compiled execution
    compiled_check = torch.compile(check_fastpath_status)
    out2 = compiled_check()

    # Verify consistency (mirroring the comparison logic in the original issue)
    assert out1 == out2, (
        f"Discrepancy between eager and compiled execution: "
        f"eager={out1}, compiled={out2}"
    )
    
    # Verify the value is actually what we set it to
    assert out1 == (not original_state), "Eager execution did not reflect the set state"

    print("Test passed: get_fastpath_enabled is consistent between eager and compiled modes.")

finally:
    # Restore original state
    set_fastpath_enabled(original_state)