```python
import os
from pathlib import Path

import tensorflow as tf
from transformers import TFAutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
# Conversion: Use TFAutoModelForCausalLM to load the TensorFlow version of the model
model = TFAutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it")


messages = [
    {"role": "user", "content": "Who are you?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    # Conversion: Use "tf" to return TensorFlow tensors instead of PyTorch tensors
    return_tensors="tf",
)
# Conversion: Device placement is handled automatically in TensorFlow, no .to() needed


example_inputs = (inputs["input_ids"], inputs["attention_mask"])

# Conversion: TensorFlow models are typically exported using model.save()
# to the SavedModel format. This replaces torch.onnx.export.
model.save("gemma3_tf")

print("TensorFlow model exported to gemma3_tf")
```