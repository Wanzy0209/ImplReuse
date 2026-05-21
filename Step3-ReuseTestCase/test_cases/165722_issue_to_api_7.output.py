import torch
import torch._dynamo
import gc
import sys

def test_dynamo_guard_dangling_pointer():
    """
    Test case for Issue 165722: RelationalGuard classes store raw PyObject* 
    without proper reference counting.
    
    This test reproduces the scenario where a guard stores a raw pointer to a tensor,
    torch._dynamo.reset() is called (clearing caches), and the tensor is deleted.
    Subsequent access to the guard should not crash, even if the pointer was dangling.
    """
    
    # Leverage the similar API (torch.backends.cuda.is_built) to determine 
    # the device for the test, ensuring the test runs in the available environment.
    if torch.backends.cuda.is_built():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Running test on device: {device}")

    # Define a simple function that triggers guards
    def simple_func(x):
        return x + 1

    # Compile the function using torch.compile (torch._dynamo)
    # This creates the guards that will store the PyObject*
    compiled_func = torch.compile(simple_func)

    # Create a tensor
    # We use a specific device based on the similar API check
    t = torch.randn(10, device=device)

    # 1. First run: This triggers the guard creation.
    # The guard (e.g., OBJECT_ALIASING) stores a raw pointer to 't' 
    # without Py_INCREF (in the buggy version).
    result1 = compiled_func(t)
    
    # 2. Reset the dynamo state.
    # This clears the internal caches. If the guard relied on the cache 
    # to keep the object alive, and didn't increment the ref count itself,
    # the pointer inside the guard is now at risk of becoming dangling.
    torch._dynamo.reset()

    # 3. Explicitly delete the tensor and force garbage collection.
    # This attempts to free the memory pointed to by the raw pointer in the guard.
    # If the bug is present, the guard now holds a dangling pointer.
    del t
    gc.collect()

    # 4. Create a new tensor to pass to the function.
    t_new = torch.randn(10, device=device)

    # 5. Run the compiled function again.
    # The guard will attempt to check the new tensor against the stored pointer.
    # If the bug exists, accessing the dangling pointer (_first_tensor == value)
    # will likely cause a segmentation fault or undefined behavior.
    # We wrap this in a try-except block, though segfaults terminate the process.
    try:
        result2 = compiled_func(t_new)
        print("Test Passed: No crash detected after reset and tensor deletion.")
        assert result2.shape == (10,)
    except RuntimeError as e:
        print(f"Test Failed with RuntimeError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Test Failed with unexpected exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_dynamo_guard_dangling_pointer()