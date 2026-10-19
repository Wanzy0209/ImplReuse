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

# Conversion: Prepare inputs for the TensorFlow model signature
# In TF, we often define a tf.function to explicitly set input/output names and shapes
@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="input_ids"),
    tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="attention_mask")
])
def serving_fn(input_ids, attention_mask):
    # Conversion: Call the model and explicitly return the logits tensor
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return {"logits": outputs.logits}

# TensorFlow Export
# Conversion: tf.saved_model.save is the standard way to export TF models (equivalent to ONNX export)
# The 'signatures' argument allows us to specify input/output names like torch.onnx.export
tf.saved_model.save(
    model,
    "gemma3_tf",  # Output directory
    signatures=serving_fn
)

print("TensorFlow model exported to gemma3_tf")
```