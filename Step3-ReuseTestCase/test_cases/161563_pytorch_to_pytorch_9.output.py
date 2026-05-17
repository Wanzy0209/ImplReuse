import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.utils import prune

# Load the model (same setup as the original bug report)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it")

# Adaptation: Instead of exporting, we test the similar API (pruning) on the model.
# We select a specific layer to apply the structured pruning.
# Gemma models typically have a structure like model.model.layers[i].mlp.gate_proj
target_module = model.model.layers[0].mlp.gate_proj

# Call the similar API: torch.nn.utils.prune.ln_structured
# Prune 20% of channels with the lowest L2-norm along dimension 0
prune.ln_structured(module=target_module, name='weight', amount=0.2, n=2, dim=0)

# Verification: Check if the mask was created and applied
assert hasattr(target_module, 'weight_mask'), "Pruning mask 'weight_mask' not found in module"
assert hasattr(target_module, 'weight_orig'), "Original weight 'weight_orig' not found in module"

# Verify that the mask is indeed binary (0 or 1)
mask = target_module.weight_mask
assert torch.all((mask == 0) | (mask == 1)), "Mask contains values other than 0 and 1"

# Verify that some weights were actually pruned (set to 0 via the mask)
assert torch.any(mask == 0), "No weights were pruned (mask is all 1s)"

print("Test passed: torch.nn.utils.prune.ln_structured executed successfully on Gemma model.")