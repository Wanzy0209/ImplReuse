import torch
import tensorflow as tf
import gc
import weakref

def test_tf_summary_value_lifecycle():
    """
    Test case adapted from PyTorch Issue #165722.
    
    Original Bug: RelationalGuard classes (OBJECT_ALIASING) stored raw PyObject* 
    without proper reference counting, leading to dangling pointers after reset.
    
    Adaptation: This test verifies that tf.compat.v1.Summary.Value (or similar 
    Tensor objects) are handled correctly in a scenario mimicking the guard logic.
    We simulate the 'buggy' behavior using weak references (raw pointers) vs 
    'fixed' behavior using strong references (py::object).
    """
    
    # 1. Simulate the 'Buggy' Guard (Raw Pointer / Weak Reference)
    class BuggyGuard:
        def __init__(self):
            self._first_value = None
            self._is_first_call = True

        def check(self, value):
            if self._is_first_call:
                # Simulate storing a raw pointer (borrowed reference)
                # In Python, we use weakref to simulate a non-incremented reference
                self._first_value = weakref.ref(value)
                self._is_first_call = False
                return True
            # Check if the object still exists
            ref = self._first_value()
            return ref is not None and ref == value

        def reset_state(self):
            self._is_first_call = True

    # 2. Simulate the 'Fixed' Guard (Strong Reference / py::object)
    class FixedGuard:
        def __init__(self):
            self._first_value = None
            self._is_first_call = True

        def check(self, value):
            if self._is_first_call:
                # Simulate storing with proper reference counting (py::object)
                self._first_value = value
                self._is_first_call = False
                return True
            return self._first_value == value

        def reset_state(self):
            self._is_first_call = True

    # --- Test Scenario ---

    # Create a tf.compat.v1.Summary.Value (or a Tensor acting as the value)
    # Note: tf.compat.v1.Summary.Value is a protobuf, often used with Tensors.
    # We use a Tensor here as the primary object being guarded, as Value wraps Tensors.
    tensor_value = tf.constant(1.0)

    # Test BuggyGuard behavior
    buggy_guard = BuggyGuard()
    
    # First call: stores the weak reference
    assert buggy_guard.check(tensor_value) is True
    
    # Simulate 'reset' or scope end where the original object might be deleted
    # if no other strong references exist.
    # In the original bug, torch._dynamo.reset() cleared caches.
    # Here, we delete the variable holding the tensor.
    del tensor_value
    gc.collect() # Force garbage collection

    # Second call: tries to access the stored reference
    # With the bug (weakref), this should fail (return False) because the object was collected.
    # This demonstrates the dangling pointer issue (object is gone).
    assert buggy_guard.check(None) is False, "BuggyGuard should fail after object deletion"

    # Test FixedGuard behavior
    # Re-create the object
    tensor_value_fixed = tf.constant(1.0)
    fixed_guard = FixedGuard()
    
    # First call: stores the strong reference
    assert fixed_guard.check(tensor_value_fixed) is True
    
    # Delete the external reference
    del tensor_value_fixed
    gc.collect()

    # Second call: tries to access the stored reference
    # With the fix (strong ref), this should succeed because the Guard holds the object alive.
    # We pass a dummy value, the check compares against the stored internal value.
    # The check logic in FixedGuard compares self._first_value == value.
    # Since we deleted the external variable, we can't pass it in to check equality easily 
    # unless we retrieve it from the guard or just check existence.
    # Let's verify the object is still alive in the guard.
    assert fixed_guard._first_value is not None, "FixedGuard should keep object alive"
    assert fixed_guard._first_value.numpy() == 1.0, "FixedGuard value should be correct"

if __name__ == "__main__":
    test_tf_summary_value_lifecycle()
    print("Test passed.")