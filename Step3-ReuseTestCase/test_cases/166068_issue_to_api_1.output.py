import torch
from packaging import version
import pytest

def test_hip_version_and_backend_introspection():
    """
    Test case to verify backend introspection attributes.
    This test leverages the pattern from torch.backends.mkldnn.is_available
    to check the availability of backends, while applying the logic from
    the bug report to parse the HIP version string.
    """
    
    # Leverage similar API: torch.backends.mkldnn.is_available
    # This checks the availability of the MKL-DNN backend.
    # The similarity lies in accessing C++ backend capabilities via Python bindings.
    mkldnn_available = torch.backends.mkldnn.is_available()
    assert isinstance(mkldnn_available, bool), "MKL-DNN availability check should return a boolean"

    # Original bug reproduction logic
    # The issue reports that torch.version.hip returns a string like '6.4.43482-0f2d60242'
    # which causes packaging.version.parse to raise InvalidVersion.
    hip_version_str = torch.version.hip

    if hip_version_str is not None:
        # Attempt to parse the HIP version string.
        # If the bug is present, this will raise packaging.version.InvalidVersion.
        # If the bug is fixed (e.g., by sanitizing the version string), this should succeed.
        try:
            parsed_version = version.parse(hip_version_str)
            # If parsing succeeds, verify it is a valid Version object
            assert isinstance(parsed_version, version.Version)
        except version.InvalidVersion as e:
            # This block captures the reported bug behavior.
            # In a regression test suite, one might use pytest.raises(InvalidVersion)
            # if testing for the bug, or simply let it fail if testing for the fix.
            # Here we explicitly raise to highlight the failure condition.
            pytest.fail(f"packaging.version.parse failed on torch.version.hip: {e}")
    else:
        # If HIP is not part of the build, skip the version parsing check
        pytest.skip("HIP is not available in this PyTorch build")