import torch
import tensorflow as tf
import unittest

class TestNameScopeSideEffects(unittest.TestCase):
    def test_name_scope_does_not_modify_global_variable_scope(self):
        """
        Adapted from PyTorch Issue 167064.
        Original Bug: torch.compile calls torch.distributions.Distribution.set_default_validate_args(False)
        globally, affecting distribution ops unexpectedly.
        
        This test verifies that tf.compat.v1.name_scope does not have similar
        redundant global side effects on unrelated global state, specifically
        checking if it leaks into the variable scope (a distinct global context in TF).
        """
        # Ensure we are in a graph context to test variable scopes properly
        with tf.Graph().as_default():
            # Initialize a variable scope
            with tf.compat.v1.variable_scope("initial_scope") as scope:
                initial_scope_name = scope.name
            
            # Use name_scope (the API under test)
            # In TF 1.x, name_scope should only affect op names, not variable scope.
            with tf.compat.v1.name_scope("nested_scope"):
                # Perform an operation to ensure the scope is active
                _ = tf.constant(1.0, name="dummy_op")
            
            # Check if the variable scope was affected (it shouldn't be)
            current_scope_name = tf.compat.v1.get_variable_scope().name
            
            # If name_scope leaked into variable_scope, this would fail
            self.assertEqual(initial_scope_name, current_scope_name,
                             "tf.compat.v1.name_scope should not modify the global variable scope.")

if __name__ == '__main__':
    unittest.main()