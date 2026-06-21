import torch
import torch.utils._pytree as pytree
import unittest

class Foo:
    pass

class Bar:
    def __eq__(self, other):
        return super().__eq__(other)

    def __hash__(self):
        return 0

# Register the custom class as a constant to trigger the specific Dynamo behavior
pytree.register_constant(Bar)

class TestDynamoGuardOnTempVar(unittest.TestCase):
    def test_compile_with_constant_mutation(self):
        """
        Reproduces Issue 166900: Attempt to generate guard on Dynamo-generated temporary variable.
        
        The bug occurs when torch.compile encounters a mutation of an object's attribute
        with a dictionary containing a pytree-registered constant.
        """
        
        @torch.compile(backend="eager")
        def fn(x, obj):
            # This mutation triggers the guard generation issue in Dynamo
            obj.attr = {3: Bar()}
            return x + 1

        input_tensor = torch.ones(3)
        obj_instance = Foo()

        # Execute the compiled function. 
        # Before the fix, this would raise an error related to guard generation.
        try:
            result = fn(input_tensor, obj_instance)
        except Exception as e:
            self.fail(f"torch.compile raised an exception: {e}")

        # Verify the computation result
        expected_tensor = torch.ones(3) + 1
        self.assertTrue(torch.equal(result, expected_tensor))
        
        # Verify the side effect (mutation) occurred
        self.assertIsInstance(obj_instance.attr, dict)
        self.assertIn(3, obj_instance.attr)
        self.assertIsInstance(obj_instance.attr[3], Bar)

if __name__ == "__main__":
    unittest.main()