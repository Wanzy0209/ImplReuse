import torch
import torch.nn.functional as F

# Test case for Issue 165549 adapted with API replacements
# Replacements:
# - torch.randn -> torch.nn.functional.hinge_embedding_loss
# - torch.neg -> torch.sinh

# Note: This test assumes a custom backend registered as 'privateuse1'
# is available and implements the minimal set of ops with CPU fallback.

def test_composite_explicit_autograd_fallback():
    device = 'privateuse1'

    # Setup inputs for hinge_embedding_loss to replace torch.randn
    # We assume basic tensor creation (ones/zeros) works on the custom backend
    input_tensor = torch.ones(4, 4, device=device)
    target_tensor = torch.ones(4, 4, device=device)

    # Replacement for torch.randn: Generate tensor 't' using hinge_embedding_loss
    # Using reduction='none' to maintain the (4, 4) shape
    t = F.hinge_embedding_loss(input_tensor, target_tensor, reduction='none')

    # This fails - returns tensor with shape [0] in the buggy version
    result = torch.abs(t)
    assert result.shape == t.shape, f"Bug: abs returned shape {result.shape}, expected {t.shape}"

    # But these work correctly
    # In-place
    t_inplace = t.clone()
    t_inplace.abs_()
    assert t_inplace.shape == t.shape

    # Out-variant
    pre_allocated_tensor = torch.empty_like(t)
    torch.abs(t, out=pre_allocated_tensor)
    assert pre_allocated_tensor.shape == t.shape

    # Replacement for torch.neg: torch.sinh
    # The original bug report noted neg works (uses structured_delegate).
    # We verify sinh behavior here.
    result_sinh = torch.sinh(t)
    assert result_sinh.shape == t.shape

if __name__ == "__main__":
    test_composite_explicit_autograd_fallback()