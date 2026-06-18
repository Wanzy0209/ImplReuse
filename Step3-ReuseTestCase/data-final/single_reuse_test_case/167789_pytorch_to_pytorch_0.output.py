import sys
import torch
import unittest

class TestDynamoRecursionLimit(unittest.TestCase):
    def test_compile_respects_sys_setrecursionlimit(self):
        """
        Regression test for issue #167789.
        Verifies that torch.compile respects sys.setrecursionlimit and does not
        raise a RecursionError prematurely due to internal C-level limits.
        """
        # Store original limit to restore later
        original_limit = sys.getrecursionlimit()

        try:
            # Set a limit higher than the default (usually 1000)
            # We choose 2000 to ensure we are testing against the default limit behavior
            sys.setrecursionlimit(2000)

            def recursive_fn(x, n):
                if n == 0:
                    return x
                return recursive_fn(x, n - 1) + 1

            # Compile the function using eager backend to isolate Dynamo behavior
            compiled_fn = torch.compile(recursive_fn, backend="eager")

            # Attempt to recurse 1500 times.
            # If torch.compile ignores sys.setrecursionlimit and uses the default (1000),
            # this will raise RecursionError.
            # If it respects the set limit (2000), this will succeed.
            input_tensor = torch.ones(3)
            result = compiled_fn(input_tensor, 1500)

            # Verify the computation is correct
            expected = input_tensor + 1500
            self.assertTrue(torch.allclose(result, expected))

        finally:
            # Restore original recursion limit
            sys.setrecursionlimit(original_limit)

if __name__ == "__main__":
    unittest.main()