import torch
import torch.nn.functional as F
import gc

def test_sigmoid_dynamo_reset_reference_safety():
    """
    Test case derived from Issue 165722 (RelationalGuard dangling pointers).
    
    This test verifies that using torch.nn.functional.sigmoid within a compiled
    function does not lead to reference counting issues when torch._dynamo.reset()
    is called. The original bug involved storing raw PyObject* without Py_INCREF,
    causing dangling pointers after reset.
    """
    # Define a function using the similar API: torch.nn.functional.sigmoid
    def func(x):
        return F.sigmoid(x)

    # Compile the function with torch.compile (which uses torch._dynamo)
    compiled_func = torch.compile(func)

    # Create a tensor input
    input_tensor = torch.randn(4, 4)

    # First execution: This triggers the creation of guards.
    # In the buggy version, a guard might store a raw pointer to input_tensor
    # or internal objects without incrementing the reference count.
    output_1 = compiled_func(input_tensor)

    # Reset the Dynamo compiler.
    # Bug Scenario: If guards held raw pointers, reset_state() would reset
    # the logic flags but leave the raw pointers dangling. If the objects
    # were garbage collected, accessing these pointers later would cause a crash.
    torch._dynamo.reset()

    # Force garbage collection to attempt to free unreferenced objects.
    # If reference counting was incorrect, this might delete the object
    # that the dangling pointer refers to.
    gc.collect()

    # Re-compile and execute again.
    # If the bug is present, the guard might attempt to access the dangling pointer,
    # leading to a segmentation fault or undefined behavior.
    compiled_func = torch.compile(func)
    output_2 = compiled_func(input_tensor)

    # Verify that the outputs are consistent
    assert torch.allclose(output_1, output_2), "Outputs differ after dynamo reset"

if __name__ == "__main__":
    test_sigmoid_dynamo_reset_reference_safety()
    print("Test passed successfully.")