import torch
from transformers import AutoModelForCausalLM
import pytest

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Test requires MPS device")
def test_mps_no_grad_with_fastpath_control():
    """
    Test case to reproduce NaN values on MPS with 4D attention mask under torch.no_grad(),
    while leveraging torch.backends.mha to control the fast path execution.
    """
    device = "mps"
    model = AutoModelForCausalLM.from_pretrained("sbintuitions/tiny-lm").to(device)
    input_ids = torch.tensor([[0, 1, 0, 0], [0, 1, 2, 3]], device=device)

    # 4D attention mask that triggers the bug
    attention_mask = torch.tensor([
        [[[ True, False, False, False],
          [ True,  True, False, False],
          [False, False, False, False],
          [False, False, False, False]]],
        [[[ True, False, False, False],
          [ True,  True, False, False],
          [ True,  True,  True, False],
          [ True,  True,  True,  True]]]], device=device)

    # Store original fastpath state to restore later
    original_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

    try:
        # Scenario 1: Ensure fastpath is enabled (often default)
        torch.backends.mha.set_fastpath_enabled(True)
        with torch.no_grad():
            outputs_fastpath = model(input_ids=input_ids, attention_mask=attention_mask)
        
        assert not torch.isnan(outputs_fastpath.logits).any(), \
            "Logits contain NaN values with fastpath enabled inside torch.no_grad()"

        # Scenario 2: Disable fastpath to check if the NaN issue is specific to that kernel path
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.no_grad():
            outputs_slowpath = model(input_ids=input_ids, attention_mask=attention_mask)
            
        assert not torch.isnan(outputs_slowpath.logits).any(), \
            "Logits contain NaN values with fastpath disabled inside torch.no_grad()"

    finally:
        # Restore the original setting
        torch.backends.mha.set_fastpath_enabled(original_fastpath_enabled)