```python
import os
from pathlib import Path

import tensorflow as tf
from transformers import TFAutoModelForCausalLM, AutoTokenizer

# Conversion: Use TFAutoModelForCausalLM for TensorFlow
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
model = TFAutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it")


messages = [
    {"role": "user", "content": "Who are you?"},
]

# Conversion: Use return_tensors="tf" to get TensorFlow tensors
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="tf",
)


# Conversion: Inputs are already tensors, no need to move to device explicitly
example_inputs = (inputs["input_ids"], inputs["attention_mask"])

# TF Export
# Conversion: torch.onnx.export is replaced by tf.saved_model.save.
# The target API context (convert_to_constants_test) relates to graph freezing,
# but standard model export in TF is done via SavedModel format.
tf.saved_model.save(
    model,
    "gemma3", # Output directory
    signatures=model.call, # Export the forward pass
)

print("TensorFlow model exported to gemma3")
```