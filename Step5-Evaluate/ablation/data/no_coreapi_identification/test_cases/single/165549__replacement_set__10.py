import torch
import torch.nn.functional as F

# Assuming 'privateuse1' backend is registered and implements CPU fallback
# as described in the bug report.

def test_abs_on_custom_backend():
    device = 'privateuse1'

    # Replacement for torch.randn: torch.nn.functional.cosine_embedding_loss
    # We need inputs for the loss function.
    # Using torch.ones to create inputs since torch.randn is replaced.
    input1 = torch.ones(10, 5, device=device)
    input2 = torch.ones(10, 5, device=device)
    target = torch.ones(10, device=device)

    # Generate tensor t using the replacement API
    # Using reduction='none' to ensure t is a non-scalar tensor (shape [10])
    t = F.cosine_embedding_loss(input1, input2, target, reduction='none')

    # Original failing case: torch.abs
    # This should return a tensor with the same shape as t
    result = torch.abs(t)

    # Bug: result.shape == torch.Size([0])
    # Expected: result.shape == t.shape
    assert result.shape == t.shape, f"Bug detected: abs returned shape {result.shape}, expected {t.shape}"

    # Case that works: in-place abs
    t_inplace = t.clone()
    t_inplace.abs_()
    assert t_inplace.shape == t.shape

    # Replacement for torch.neg: torch.exp2
    # Case that works: other op (structured_delegate or similar)
    result_exp2 = torch.exp2(t)
    assert result_exp2.shape == t.shape

if __name__ == "__main__":
    test_abs_on_custom_backend()