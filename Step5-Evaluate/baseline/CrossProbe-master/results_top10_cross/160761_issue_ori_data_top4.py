```python
import os
from pathlib import Path

import tensorflow as tf
from transformers import TFAutoModelForCausalLM, AutoTokenizer

# Conversion comment: Use TFAutoModelForCausalLM for TensorFlow
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
model = TFAutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it")


messages = [
    {"role": "user", "content": "Who are you?"},
]

# Conversion comment: Change return_tensors to "tf" for TensorFlow tensors
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="tf",
)

# Conversion comment: TF tensors do not require explicit .to(device) placement like PyTorch
example_inputs = (inputs["input_ids"], inputs["attention_mask"])

# Conversion comment: Replacing torch.onnx.export with clone_and_build_model.
# This clones the model and builds it with the provided inputs, preparing it for use
# in a potentially separate graph or context.
# Note: ONNX specific parameters (opset_version, do_constant_folding, dynamo) are
# not applicable to the TensorFlow cloning API.
cloned_model = tf.keras.models.clone_and_build_model(
    model,
    input_tensors=list(example_inputs), # API expects list or dict
    in_place_reset=False
)

# Conversion comment: Save the cloned model to disk (SavedModel format)
# to fulfill the intent of the original 'export' operation.
tf.saved_model.save(cloned_model, "gemma3_tf")

print("TensorFlow model cloned and saved to gemma3_tf")
```