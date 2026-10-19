```python
import os
from pathlib import Path

import tensorflow as tf
from transformers import TFAutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
# Conversion: Use TFAutoModelForCausalLM to load the model as a TensorFlow/Keras model
model = TFAutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it")


messages = [
    {"role": "user", "content": "Who are you?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    # Conversion: Return TensorFlow tensors instead of PyTorch tensors
    return_tensors="tf",
)
# Conversion: TensorFlow manages device placement automatically; explicit .to() is not required

example_inputs = (inputs["input_ids"], inputs["attention_mask"])

# Conversion: Save the model to HDF5 format instead of exporting to ONNX.
# This saves the model's configuration, weights, and optimizer state (if available).
model.save("gemma3.h5", save_format="h5")

print("Model saved to gemma3.h5")
```