import torch
import logging
import pytest

def test_nnpack_logging_format_mismatch():
    """
    Test case to reproduce the logging format error similar to Issue 166630.
    
    The original bug occurred in torch.distributed.launcher.api.py where 
    logger.info was called with multiple string arguments followed by a 
    dictionary, causing a mismatch between the format string placeholders 
    and the arguments tuple.
    
    This test leverages torch.backends.nnpack.set_flags (the similar API)
    to generate data and attempts to log it using the same buggy pattern,
    verifying that the pattern indeed causes an error.
    """
    # Check if nnpack is available
    if not hasattr(torch.backends, 'nnpack'):
        pytest.skip("nnpack backend not available")

    logger = logging.getLogger("test_nnpack")
    logger.setLevel(logging.INFO)

    # Use the similar API to generate data
    # set_flags returns the original flags as a tuple
    try:
        original_flags = torch.backends.nnpack.set_flags(True)
    except Exception:
        pytest.skip("nnpack set_flags failed")

    # Restore state to avoid side effects
    torch.backends.nnpack.set_flags(original_flags[0])

    # Reproduce the bug logic:
    # The logger.info call is structured with split strings and a dict at the end.
    # This causes the format operator to fail because the first string contains
    # named placeholders (e.g., %(enabled)s), but the args passed to the formatter
    # are a tuple (str, str, dict) instead of a single mapping.
    with pytest.raises((TypeError, KeyError)):
        logger.info(
            "  nnpack_enabled          : %(enabled)s\n",
            "  returned_flags          : %(flags)s\n",
            {
                "enabled": True,
                "flags": original_flags
            }
        )