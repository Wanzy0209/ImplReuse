import torch

def test_torch_all_with_compile():
    """
    Test case for torch.all in the context of torch.compile.
    Related to Issue 167064 which discusses side effects of torch.compile initialization.
    """
    
    def fn(x):
        return torch.all(x)

    # Compile the function using torch.compile
    # This mimics the context where the bug (redundant global code triggering compile) occurs.
    compiled_fn = torch.compile(fn)

    # Test case 1: All elements are True
    input_tensor = torch.tensor([True, True, True])
    result = compiled_fn(input_tensor)
    assert result.item() == True, "torch.all failed on all-True tensor"

    # Test case 2: Some elements are False
    input_tensor = torch.tensor([True, False, True])
    result = compiled_fn(input_tensor)
    assert result.item() == False, "torch.all failed on mixed tensor"

    # Test case 3: Empty tensor (should return True)
    input_tensor = torch.tensor([])
    result = compiled_fn(input_tensor)
    assert result.item() == True, "torch.all failed on empty tensor"

    print("Test passed.")

if __name__ == "__main__":
    test_torch_all_with_compile()