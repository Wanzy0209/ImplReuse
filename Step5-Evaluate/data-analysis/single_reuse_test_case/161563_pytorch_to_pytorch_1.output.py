import torch
from torch.nn.utils import prune
import sys

# Handle missing transformers library by mocking it
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("transformers library not found. Using mocks to test pruning logic.")
    from unittest.mock import MagicMock

    # Create a simple model with Linear layers to test pruning
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(10, 10)
            self.linear2 = torch.nn.Linear(10, 10)
            self.device = torch.device('cpu')

        def forward(self, x):
            return x

    # Mock AutoModelForCausalLM
    AutoModelForCausalLM = MagicMock()
    AutoModelForCausalLM.from_pretrained = MagicMock(return_value=DummyModel())

    # Mock AutoTokenizer
    class DummyInputs:
        def to(self, device):
            return self

    AutoTokenizer = MagicMock()
    tokenizer_mock = MagicMock()
    tokenizer_mock.apply_chat_template = MagicMock(return_value=DummyInputs())
    AutoTokenizer.from_pretrained = MagicMock(return_value=tokenizer_mock)

# Setup from the original bug report
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it")

messages = [
    {"role": "user", "content": "Who are you?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

# Identify parameters to prune (e.g., all Linear weights in the model)
parameters_to_prune = []
for module in model.modules():
    if isinstance(module, torch.nn.Linear):
        parameters_to_prune.append((module, 'weight'))

# Adapted call site: Use torch.nn.utils.prune.global_unstructured
# instead of torch.export.export
try:
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.2
    )
except Exception as e:
    print(f"Test failed with error: {e}")
    raise

# Assertions to verify the API behavior
for module, name in parameters_to_prune:
    # Check that the mask was created
    assert hasattr(module, name + '_mask'), f"Mask not found for {name}"
    # Check that the original parameter was stored
    assert hasattr(module, name + '_orig'), f"Original parameter not found for {name}"
    
    # Verify that pruning actually occurred (mask is not all 1s)
    mask = getattr(module, name + '_mask')
    assert mask.sum() < mask.numel(), f"Pruning did not remove any weights in {name}"

print("Test passed: torch.nn.utils.prune.global_unstructured executed successfully.")