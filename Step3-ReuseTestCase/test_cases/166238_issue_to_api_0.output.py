import torch
import collections
import torch.nn.functional as F

def test_dynamo_defaultdict_with_hardtanh():
    """
    Test case for Issue 166238: Regression about collections.defaultdict creation.
    This test preserves the original bug reproduction logic (creating a defaultdict
    inside a compiled function) while leveraging torch.nn.functional.hardtanh_
    as the similar API to ensure interaction between standard ops and containers.
    """
    
    def func(x):
        # Leverage the similar API: torch.nn.functional.hardtanh_
        # This operation is supported and serves as a baseline for the compilation context.
        F.hardtanh_(x, min_val=-1.0, max_val=1.0)
        
        # Original bug reproduction logic: Creating collections.defaultdict
        # Dynamo previously raised an Unsupported error for this call.
        # The error context involved UserDefinedClassVariable and default_factory.
        dd = collections.defaultdict(list)
        dd["result"].append(x.sum().item())
        
        return x, dd

    # Compile the function with torch.compile
    compiled_func = torch.compile(func)
    
    # Create a dummy input tensor
    input_tensor = torch.randn(3, 3)
    
    # Execute the compiled function
    # If the bug is present, this will raise:
    # torch._dynamo.exc.Unsupported: Dynamo does not know how to trace the function `<class 'collections.defaultdict'>`
    result_tensor, result_dict = compiled_func(input_tensor)
    
    # Assertions to verify correct behavior
    assert isinstance(result_dict, collections.defaultdict)
    assert "result" in result_dict
    assert len(result_dict["result"]) == 1
    
    # Verify the hardtanh operation was applied correctly
    expected_tensor = input_tensor.clone()
    F.hardtanh_(expected_tensor, min_val=-1.0, max_val=1.0)
    assert torch.allclose(result_tensor, expected_tensor)

if __name__ == "__main__":
    test_dynamo_defaultdict_with_hardtanh()
    print("Test passed.")