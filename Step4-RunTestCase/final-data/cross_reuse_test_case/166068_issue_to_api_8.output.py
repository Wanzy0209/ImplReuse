import torch
from packaging import version
import pytest

def test_torch_version_hip_format():
    """
    Test case for Issue 166068.
    
    Verifies that torch.version.hip adheres to PEP 440 versioning standards
    expected by packaging.version.parse.
    
    The bug report indicates that torch.version.hip returns a string like
    '6.4.43482-0f2d60242' which includes a git hash, causing InvalidVersion error.
    In contrast, torch.version.cuda returns a standard version like '12.6'.
    """
    # Only run this test if HIP is available
    if not torch.version.hip:
        pytest.skip("HIP is not available in this build")

    # The original bug reproduction logic
    # This expects an InvalidVersion to be raised due to the non-standard format
    with pytest.raises(version.InvalidVersion):
        version.parse(torch.version.hip)

def test_torch_version_cuda_format():
    """
    Control test to verify that torch.version.cuda parses correctly.
    """
    if not torch.version.cuda:
        pytest.skip("CUDA is not available in this build")

    # This should succeed without raising an exception
    parsed_version = version.parse(torch.version.cuda)
    assert parsed_version is not None