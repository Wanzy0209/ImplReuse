import torch
from torch.distributions.constraints import greater_than_eq

def test_greater_than_eq_mps_comparison():
    """
    Test case for torch.distributions.constraints.greater_than_eq based on 
    Issue 167767 (clamp incorrectness with mps backend).
    
    The original issue demonstrates that torch.clamp(min=1e-7) fails to update
    a tensor containing 0.0 on the MPS backend. This implies a potential issue
    with the comparison logic (0.0 < 1e-7) or the application of the lower bound.
    
    This test verifies that the greater_than_eq constraint, which relies on the
    same comparison logic (lower_bound <= value), correctly identifies that
    0.0 is NOT greater than or equal to 1e-7 on the MPS backend.
    """
    
    if not torch.backends.mps.is_available():
        print("MPS backend is not available. Skipping test.")
        return

    # Reproduce the setup from the original bug report
    # The original issue notes that this specific line triggers incorrect behavior
    a = torch.zeros(1, device='mps')
    a_clamped = a.clamp(min=0.0)

    # Define the constraint with the same lower bound used in the failing clamp calls
    lower_bound = 1e-7
    constraint = greater_than_eq(lower_bound)

    # Create the tensor to be checked, mirroring the 'b' tensor in the issue
    b = torch.zeros(1, device='mps')
    
    # Perform the check. 
    # Logic: lower_bound (1e-7) <= value (0.0)
    # Expected Result: False (because 0.0 is less than 1e-7)
    # If the MPS backend bug affects comparisons, this might incorrectly return True.
    result = constraint.check(b)

    print(f"Input Tensor: {b.item()}")
    print(f"Constraint: {constraint}")
    print(f"Check Result (Expected False): {result.item()}")

    # Assertion to catch the regression
    assert not result.item(), (
        f"Bug detected: greater_than_eq check incorrectly returned True. "
        f"Value {b.item()} is not >= {lower_bound}."
    )

if __name__ == "__main__":
    test_greater_than_eq_mps_comparison()