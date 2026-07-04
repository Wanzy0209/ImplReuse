# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_isolated_language_model_mps(device):
    model = AutoModelForCausalLM.from_pretrained("sbintuitions/tiny-lm").to(device)
    input_ids = torch.tensor([[0, 1, 0, 0], [0, 1, 2, 3]], device=device)

    attention_mask = torch.tensor([
        [[[ True, False, False, False],
          [ True,  True, False, False],
          [False, False, False, False],
          [False, False, False, False]]],
        [[[ True, False, False, False],
          [ True,  True, False, False],
          [ True,  True,  True, False],
          [ True,  True,  True,  True]]]], device=device)
    # attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool, device=device)

    with torch.no_grad():
       outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    assert not torch.isnan(outputs.logits).any(), "Logits contain NaN values"