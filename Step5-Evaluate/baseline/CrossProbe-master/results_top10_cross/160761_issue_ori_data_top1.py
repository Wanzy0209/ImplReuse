```python
import os
from pathlib import Path

import tensorflow as tf
from transformers import TFAutoModelForCausalLM, AutoTokenizer
# Conversion comment: The target API context suggests using convert_to_constants 
# to handle variable freezing, similar to do_constant_folding=True in ONNX export.
from tensorflow.python.framework import convert_to_constants

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
    return_tensors="tf", # Conversion: Changed from "pt" to "tf" for TensorFlow
)
# Conversion: TensorFlow handles device placement automatically, explicit .to() is not needed here.

# Conversion: Prepare inputs as a tuple to match the source structure
example_inputs = (inputs["input_ids"], inputs["attention_mask"])

# Export
# Conversion: torch.onnx.export is translated to tf.saved_model.save.
# To match the 'do_constant_folding=True' and the provided context API, 
# we convert variables to constants (freeze the graph) before saving.

# Get the concrete function of the model call
@tf.function
def model_call(input_ids, attention_mask):
    return model(input_ids=input_ids, attention_mask=attention_mask)

concrete_function = model_call.get_concrete_function(
    input_ids=tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="input_ids"),
    attention_mask=tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="attention_mask")
)

# Convert variables to constants (Freezing)
# This aligns with the 'convert_to_constants' context and 'do_constant_folding=True'
frozen_model = convert_to_constants.convert_variables_to_constants_v2(concrete_function)

# Save the frozen model
tf.saved_model.save(frozen_model, "gemma3_tf")

print("TensorFlow model exported to gemma3_tf")
```