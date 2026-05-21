import pytest
import torch
from packaging import version
import tensorflow as tf

def test_torch_hip_version_strict_parsing():
    """
    Test case to reproduce the InvalidVersion error for torch.version.hip.
    Leverages tf.experimental.enable_strict_mode to align the test context
    with the concept of strictness found in the similar API.
    """
    # Leverage the similar API to establish a strict testing environment
    tf.experimental.enable_strict_mode()

    # Original bug reproduction logic
    # The version string '6.4.43482-0f2d60242' is not PEP 440 compliant
    hip_version = torch.version.hip

    # Assert that parsing the HIP version raises an InvalidVersion error
    with pytest.raises(version.InvalidVersion):
        version.parse(hip_version)