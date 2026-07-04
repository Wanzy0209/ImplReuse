import pytest
import torch
from transformers import AutoModelForCausalLM

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS device not available")
def test_no_grad_mps_4d_mask_nan():
    """
    Test to verify that torch.no_grad() does not yield NaN values on MPS device
    when using a 4D attention mask.
    
    This test reproduces the issue where specific 4D attention masks combined
    with the no_grad context manager on the MPS backend result in NaN logits.
    """
    device = "mps"
    
    # Load a small causal LM for testing
    model = AutoModelForCausalLM.from_pretrained("sbintuitions/tiny-lm").to(device)
    model.eval()  # Ensure model is in evaluation mode

    # Define input IDs
    input_ids = torch.tensor([[0, 1, 0, 0], [0, 1, 2, 3]], device=device)

    # Define the problematic 4D attention mask
    # This specific structure (padded/irregular) triggers the NaN issue on MPS
    attention_mask = torch.tensor([
        [[[ True, False, False, False],
          [ True,  True, False, False],
          [False, False, False, False],
          [False, False, False, False]]],
        [[[ True, False, False, False],
          [ True,  True, False, False],
          [ True,  True,  True, False],
          [ True,  True,  True,  True]]]], device=device)

    # Run the model inside torch.no_grad()
    # This context manager is the specific trigger for the bug on MPS
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Assert that the output logits do not contain NaN values
    assert not torch.isnan(outputs.logits).any(), "Logits contain NaN values inside torch.no_grad() on MPS"