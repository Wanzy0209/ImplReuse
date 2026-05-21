import torch
import tensorflow as tf
from tensorflow.python.compat import compat

def test_forward_compatibility_horizon_access_and_conversion():
    """
    Tests the stability of accessing and using forward_compatibility_horizon.
    This mirrors the PyTorch issue where accessing OperatorExportTypes and
    triggering a conversion (__int__) caused a segfault during import.
    """
    # 1. Access the API (mimicking the import/access phase of the bug)
    # In the PyTorch bug, accessing torch._C._onnx.OperatorExportTypes was the trigger.
    api = compat.forward_compatibility_horizon

    # 2. Invoke the API with integer arguments.
    # The PyTorch backtrace shows a crash in __int__ conversion.
    # Here we pass integers (year, month, day) to ensure the API handles
    # integer arguments correctly without memory issues.
    year, month, day = 2018, 8, 1
    context_manager = api(year, month, day)

    # 3. Verify the object is valid and usable.
    # The PyTorch bug resulted in an unusable object (segfault).
    # We assert that the returned object is a proper context manager.
    assert hasattr(context_manager, '__enter__'), "Returned object must be a context manager"
    assert hasattr(context_manager, '__exit__'), "Returned object must be a context manager"

    # 4. Execute the lifecycle.
    # Ensures no hidden errors occur during the 'use' phase.
    with context_manager:
        # The context is active here.
        pass