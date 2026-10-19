```python
import os
from pathlib import Path

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM
# Conversion: Import the specific utility for casting inputs to model dtypes
from tensorflow.python.keras.engine.training_utils_v1 import cast_to_model_input_dtypes

# Conversion: Use TFAutoModelForCausalLM for TensorFlow
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
model = TFAutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it")


messages = [
    {"role": "user", "content": "Who are you?"},
]

# Conversion: Change return_tensors to "tf" for TensorFlow tensors
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="tf",
)
# Conversion: TensorFlow handles device placement automatically, no .to(model.device) needed


# Conversion: Prepare inputs as a tuple of tensors
raw_inputs = (inputs["input_ids"], inputs["attention_mask"])

# Conversion: Use the target API to cast inputs to match the model's expected dtypes
# This replaces the explicit device/type handling often required in PyTorch
example_inputs = cast_to_model_input_dtypes(raw_inputs, model)

# Conversion: TensorFlow equivalent of exporting a model is saving it as a SavedModel
# Note: Direct ONNX export in TensorFlow typically requires the tf2onnx library.
# Here we perform the native TensorFlow export (saving the model).
model.save("gemma3_tf")

print("TensorFlow model exported to gemma3_tf")
```