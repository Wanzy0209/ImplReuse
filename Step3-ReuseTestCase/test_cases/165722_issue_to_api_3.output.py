import torch
import tensorflow as tf
import pytest

def test_keras_backend_clear_session_dangling_reference():
    """
    Test case adapted from PyTorch Issue #165722.
    
    Original Bug Logic:
    1. RelationalGuard stores a raw PyObject* (_first_tensor) without Py_INCREF.
    2. torch._dynamo.reset() is called, potentially deleting the object.
    3. Subsequent access to _first_tensor causes a dangling pointer crash.
    
    Similar API Logic (tf.keras.backend):
    1. Create a symbolic tensor managed by the backend session.
    2. Call tf.keras.backend.clear_session() (analogous to reset).
    3. Attempt to access the tensor.
    
    Expected Behavior:
    The system should handle the reset safely. Accessing the old object should
    raise a defined error (e.g., ValueError) rather than causing a segmentation fault,
    confirming that reference counting or invalidation logic is correct.
    """
    # 1. Create a symbolic tensor (analogous to the PyObject* stored in the guard)
    # This object is tied to the current Keras backend session/graph.
    input_tensor = tf.keras.Input(shape=(32,), name="test_input")
    
    # 2. Perform the reset operation
    # tf.keras.backend.clear_session() destroys the current TF graph and resets the session.
    # This is the semantic equivalent of torch._dynamo.reset().
    tf.keras.backend.clear_session()
    
    # 3. Attempt to access the object after the reset
    # In the PyTorch bug, this accesses a dangling pointer (Use-After-Free).
    # In Keras, this should raise a safe error indicating the tensor is invalid
    # or belongs to a destroyed graph, rather than crashing the interpreter.
    with pytest.raises((ValueError, RuntimeError, AttributeError)):
        # Trying to use the tensor from the cleared session
        # typically raises: "Tensor ... was created in a different session"
        _ = input_tensor + 1