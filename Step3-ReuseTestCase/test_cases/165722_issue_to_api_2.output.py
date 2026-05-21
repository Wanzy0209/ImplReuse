import torch
import tensorflow as tf
from tensorflow.keras import backend as K

# This test case mimics the logic of the PyTorch bug (Issue 165722).
# The bug involved a C++ class (OBJECT_ALIASING) storing a raw PyObject* 
# without reference counting, which became a dangling pointer after 
# torch._dynamo.reset().
#
# Here, we adapt this logic to the similar API: tf.keras.backend.epsilon.
# We simulate the 'Guard' pattern in Python, storing the result of epsilon(),
# resetting the global state (analogous to torch._dynamo.reset), and then
# accessing the stored value again to ensure stability.

class EpsilonAliasGuard:
    """
    Mimics the OBJECT_ALIASING class from torch/csrc/dynamo/guards.cpp.
    Stores the first observed value of epsilon and compares subsequent calls to it.
    """
    def __init__(self):
        self._is_first_call = True
        self._first_epsilon = None

    def check(self):
        # Get the current value from the similar API
        current_epsilon = K.epsilon()
        
        if self._is_first_call:
            # In the C++ bug, this was: _first_tensor = value; (without Py_INCREF)
            # In Python, this creates a proper reference.
            self._first_epsilon = current_epsilon
            self._is_first_call = False
            return True
        
        # Check if the current value matches the stored value
        return self._first_epsilon == current_epsilon

def test_epsilon_guard_with_reset():
    """
    Test that verifies the stability of tf.keras.backend.epsilon 
    when used in a stateful guard pattern across a global session reset.
    """
    guard = EpsilonAliasGuard()

    # 1. First call: stores the reference
    assert guard.check() is True, "First call should return True"
    
    # 2. Reset the global state
    # Analogous to torch._dynamo.reset() in the original bug report.
    # This clears the Keras session, potentially invalidating internal states.
    K.clear_session()

    # 3. Second call: accesses the stored reference
    # In the original bug, accessing the stored pointer after reset caused a crash
    # (dangling pointer). Here, we verify that the API handles the state transition
    # gracefully and the logic remains consistent.
    try:
        result = guard.check()
        # We expect True because epsilon is a global config usually persistent 
        # or reset to a default, but primarily we expect no crash.
        assert result is True, "Epsilon value should remain consistent after reset"
    except Exception as e:
        raise AssertionError(f"Accessing epsilon after reset failed with: {e}")

if __name__ == "__main__":
    test_epsilon_guard_with_reset()
    print("Test passed.")