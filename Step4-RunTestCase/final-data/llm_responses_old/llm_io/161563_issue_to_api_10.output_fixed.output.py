import torch
import pytest

def test_export_gemma_current_active_mode_not_registered():
    """
    Test case for Issue 161563: [export] "Current active mode not registered" when exporting vmap.
    
    This test reproduces the bug where torch.export.export fails with an AssertionError
    when exporting the google/gemma-3-270m-it model. The error message indicates
    an issue with the active mode not being registered.
    """
    # Moved importorskip inside the function to prevent module-level crash
    # if transformers is not installed.
    transformers = pytest.importorskip("transformers")

    model_name = "google/gemma-3-270m-it"
    
    # Load model and tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)

    # Prepare inputs using the chat template
    messages = [{"role": "user", "content": "Who are you?"}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    example_inputs = (inputs["input_ids"], inputs["attention_mask"])

    # The bug report indicates that torch.export.export raises an AssertionError
    # with the message "Current active mode ... not registered".
    # We assert this specific behavior to confirm the reproduction of the issue.
    with pytest.raises(AssertionError, match="Current active mode.*not registered"):
        torch.export.export(model, example_inputs)