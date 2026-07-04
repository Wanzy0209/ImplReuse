import torch

def test_torch_all_with_aot_eager():
    """
    Test case for torch.all within the context of torch.compile(aot_eager).
    Verifies that torch.all returns the correct boolean result even when
    compiled, ensuring the logic is not optimized away incorrectly.
    """
    def func_all_false():
        a = torch.tensor([1.0, -2.0], device="cuda")
        # Similar API: torch.all
        return torch.all(a > 0)

    def func_all_true():
        a = torch.tensor([1.0, 2.0], device="cuda")
        # Similar API: torch.all
        return torch.all(a > 0)

    # Fix: Check if _dynamo exists before attempting to reset it.
    # This handles environments where the internal API might not be exposed.
    if hasattr(torch, '_dynamo'):
        torch._dynamo.reset()
    
    # Test case where result should be False
    f_c_false = torch.compile(func_all_false, backend="aot_eager")
    result_false = f_c_false()
    assert result_false == False, "torch.all should return False when not all elements meet the condition"

    # Test case where result should be True
    f_c_true = torch.compile(func_all_true, backend="aot_eager")
    result_true = f_c_true()
    assert result_true == True, "torch.all should return True when all elements meet the condition"

if __name__ == "__main__":
    test_torch_all_with_aot_eager()