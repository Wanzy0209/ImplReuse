import torch
import gc
import sys

def test_itt_range_pop_reference_counting():
    """
    Test case for torch.profiler.itt.range_pop based on Issue 165722.
    
    Issue 165722 describes a bug where RelationalGuard classes store raw PyObject*
    without proper reference counting (Py_INCREF), leading to dangling pointers
    after state resets or object deletion.
    
    This test adapts that logic to torch.profiler.itt.range_pop. If the ITT
    implementation stores the range name (PyObject*) without incrementing the
    reference count, deleting the Python object before popping the range
    could cause a segmentation fault or undefined behavior.
    """
    
    # Test Case 1: Single Range Lifecycle
    # Create a range name and push it to the ITT stack
    range_name = "test_scope_1"
    torch.profiler.itt.range_push(range_name)
    
    # Delete the reference to the string object and force garbage collection.
    # If the C++ backend stored a raw PyObject* without Py_INCREF,
    # the pointer inside the ITT stack is now dangling.
    del range_name
    gc.collect()
    
    # Attempt to pop the range. This should not crash.
    # If the bug exists, this line might trigger a segfault.
    depth = torch.profiler.itt.range_pop()
    assert depth == 0, "Expected stack depth to be 0 after popping the only range"

    # Test Case 2: Nested Ranges Lifecycle
    # Create nested ranges and delete their Python references
    outer_name = "outer_scope"
    inner_name = "inner_scope"
    
    torch.profiler.itt.range_push(outer_name)
    torch.profiler.itt.range_push(inner_name)
    
    # Invalidate Python objects
    del outer_name, inner_name
    gc.collect()
    
    # Pop inner range
    depth = torch.profiler.itt.range_pop()
    assert depth == 1, "Expected stack depth to be 1 after popping inner range"
    
    # Pop outer range
    depth = torch.profiler.itt.range_pop()
    assert depth == 0, "Expected stack depth to be 0 after popping outer range"

    print("Test passed: torch.profiler.itt handles object lifecycles correctly.")

if __name__ == "__main__":
    test_itt_range_pop_reference_counting()