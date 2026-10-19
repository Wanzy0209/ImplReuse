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

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    # Conversion: Change return_tensors to "tf" for TensorFlow tensors
    return_tensors="tf",
)
# Conversion: .to(device) is not applicable for TF tensors in this context

example_inputs = (inputs["input_ids"], inputs["attention_mask"])

# TF Export (SavedModel)
# Conversion: torch.onnx.export is replaced by tf.saved_model.save
# Note: ONNX specific arguments (opset_version, do_constant_folding, dynamo) are omitted as they are PyTorch specific.
# TensorFlow handles graph optimization and constant folding differently.
tf.saved_model.save(
    model,
    "gemma3_tf",
)

print("TensorFlow model exported to gemma3_tf")
```