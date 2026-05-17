import torch
import torch.utils.cpp_extension
from unittest.mock import patch

def test_dynamo_codegen_with_is_ninja_available():
    """
    Test case that reproduces the 'local and cell with same name' bug
    while leveraging torch.utils.cpp_extension.is_ninja_available.
    
    This verifies that torch.compile handles variable scoping correctly
    even when calling utility functions like is_ninja_available.
    """
    
    # Mock subprocess.check_output to ensure is_ninja_available 
    # returns True deterministically without requiring ninja to be installed.
    with patch('subprocess.check_output', return_value=b'1.10.0'):
        keys = range(10)
        allowed = [0, 1, 2, 3]

        def fn(x):
            x = x + 1
            torch._dynamo.graph_break()
            
            # Original bug reproduction logic:
            # 'key' is used in the list comprehension (local) and 
            # captured as a cell variable by the inner function.
            key = [key for key in keys if key in allowed]

            def inner():
                nonlocal key

            # Leveraging the similar API: is_ninja_available
            # We call it here to ensure it interacts correctly with the 
            # complex scoping logic above during compilation.
            is_ninja = torch.utils.cpp_extension.is_ninja_available()

            # Return a value based on the scope and the API call
            return x + key[0] + (1 if is_ninja else 0)

        # Compile the function with the eager backend
        compiled_fn = torch.compile(fn, backend="eager")
        
        # Execute with a sample tensor
        input_tensor = torch.ones(3)
        result = compiled_fn(input_tensor)

        # Assertions
        # x starts as 1.0. x + 1 = 2.0.
        # key[0] is 0.
        # is_ninja is True (mocked), so + 1.
        # Expected: 2.0 + 0 + 1 = 3.0
        expected = torch.ones(3) * 3.0
        assert torch.equal(result, expected), f"Expected {expected}, got {result}"
        
        print("Test passed: torch.compile handled scope conflict with is_ninja_available correctly.")

if __name__ == "__main__":
    test_dynamo_codegen_with_is_ninja_available()