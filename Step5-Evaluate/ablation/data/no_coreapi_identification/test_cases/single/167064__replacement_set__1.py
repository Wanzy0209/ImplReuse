import torch

def test_eq_side_effect_simulation():
    """
    Test case generated based on Issue 167064.
    
    Original Bug Logic:
    - Action: torch.compile (redundant global code)
    - Side Effect: torch.distributions.Distribution.set_default_validate_args(False)
    - Impact: Affects distribution ops.
    
    Translated Logic (API Replacements):
    - Action: torch.eq (replaces torch.compile)
    - Check/Assertion: torch.eq (replaces set_default_validate_args)
    
    This test verifies the behavior of torch.eq in a structure similar to the 
    reported bug, ensuring the API interaction is valid.
    """
    # Setup inputs for the operation
    # In the original bug, this was the context of the parallel attention code.
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([1, 2, 3])

    # Original: torch.compile triggers the side effect.
    # Replacement: torch.eq is called as the primary action.
    # We simulate the "redundant global code" call.
    _ = torch.eq(x, y)

    # Original: torch.distributions.Distribution.set_default_validate_args(False) is called.
    # This was the unwanted side effect/state change.
    # Replacement: We use torch.eq to check the state/result.
    # We verify that the result of the action matches the expected state (True).
    
    result = torch.eq(x, y)
    expected = torch.tensor([True, True, True])

    # Original: Implicitly checking if the global state was set to False.
    # Replacement: Explicitly checking if the result equals the expected value using torch.eq.
    # This preserves the "Action -> Check" pattern of the bug report.
    assert torch.eq(result, expected).all().item()

if __name__ == "__main__":
    test_eq_side_effect_simulation()