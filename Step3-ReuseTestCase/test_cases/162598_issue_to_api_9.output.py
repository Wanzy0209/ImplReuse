import torch
import pytest

def test_torch_backends_nnpack_is_available():
    """
    Test case for torch.backends.nnpack.is_available.
    
    Context: The related issue (162598) describes a failure in a workflow 
    due to missing artifacts (resources not found). The similar API 
    'torch.backends.nnpack.is_available' checks for the presence of a 
    specific backend resource. Unlike the failing workflow which crashed 
    when resources were missing, this API should gracefully return a boolean 
    indicating the status.
    """
    # Call the API to check availability
    is_available = torch.backends.nnpack.is_available()
    
    # Assert that the function returns a boolean value
    assert isinstance(is_available, bool), \
        f"torch.backends.nnpack.is_available() should return a boolean, got {type(is_available)}"
    
    # If the backend is available, we can optionally check related flags,
    # but the primary test is ensuring the availability check itself does not crash,
    # addressing the "missing resource" concern from the bug report.
    if is_available:
        # If available, ensure the flag can be toggled (optional sanity check)
        original_state = torch.backends.nnpack.enabled
        torch.backends.nnpack.enabled = False
        assert torch.backends.nnpack.enabled is False
        torch.backends.nnpack.enabled = original_state