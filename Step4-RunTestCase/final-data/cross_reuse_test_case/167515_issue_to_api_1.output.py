import pytest
import torch
from transformers import AutoModelForCausalLM

# Leveraging the similar API structure (tf.compat.v1.train.SessionRunValues)
# to encapsulate the execution results and metadata for verification.
class SessionRunValues:
    """
    Contains the results of a model run, mimicking the structure of
    tf.compat.v1.train.SessionRunValues to organize test assertions.
    """
    def __init__(self, results, options=None, run_metadata=None):
        self.results = results
        self.options = options
        self.run_metadata = run_metadata

@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_no_grad_mps_4d_mask_nan(device):
    """
    Test to reproduce Issue 167515: torch.no_grad() yields NaN values on mps device with 4D attention mask.
    This test preserves the original bug logic while reusing the SessionRunValues pattern
    to structure the output verification.
    """
    # Skip MPS if not available (e.g., on non-Mac hardware)
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS device not available")

    model = AutoModelForCausalLM.from_pretrained("sbintuitions/tiny-lm").to(device)
    input_ids = torch.tensor([[0, 1, 0, 0], [0, 1, 2, 3]], device=device)

    # 4D attention mask as described in the bug report
    attention_mask = torch.tensor([
        [[[ True, False, False, False],
          [ True,  True, False, False],
          [False, False, False, False],
          [False, False, False, False]]],
        [[[ True, False, False, False],
          [ True,  True, False, False],
          [ True,  True,  True, False],
          [ True,  True,  True,  True]]]], device=device)

    # The bug occurs specifically inside torch.no_grad()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Wrap outputs in the reused API structure
    run_values = SessionRunValues(
        results=outputs,
        options={"device": device, "no_grad": True}
    )

    # Assert on the results field, similar to how one might inspect SessionRunValues.results
    # This checks for the NaN issue reported in the bug.
    assert not torch.isnan(run_values.results.logits).any(), \
        f"Logits contain NaN values on {device} with 4D mask and no_grad"