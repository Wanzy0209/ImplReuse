```python
import os
from pathlib import Path

import tensorflow as tf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tensorflow.python.ops.autograph_ops import wrap_py_func

# Conversion comment: Loading the model in PyTorch as the source logic relies on torch.onnx.export
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

# Conversion comment: Convert PyTorch inputs to TensorFlow tensors to pass into the TF graph
tf_input_ids = tf.constant(inputs["input_ids"].numpy())
tf_attention_mask = tf.constant(inputs["attention_mask"].numpy())

# Conversion comment: Define the function to be wrapped. 
# It receives numpy arrays (from TF tensors), converts them back to torch tensors, and exports.
def export_onnx(input_ids, attention_mask):
    # Convert numpy inputs back to PyTorch tensors
    torch_input_ids = torch.from_numpy(input_ids).to(model.device)
    torch_attention_mask = torch.from_numpy(attention_mask).to(model.device)

    example_inputs = (torch_input_ids, torch_attention_mask)

    # ONNX Export
    torch.onnx.export(
        model,
        example_inputs,
        "gemma3.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=True,
    )
    
    # Return a dummy tensor to satisfy TensorFlow's requirement that wrapped funcs return tensors
    return tf.constant(1.0)

# Conversion comment: Use wrap_py_func to integrate the PyTorch ONNX export into a TensorFlow operation
# This allows the export to be part of a TF graph or executed in a TF context
export_op = wrap_py_func(
    export_onnx,
    args=[tf_input_ids, tf_attention_mask],
)

# Execute the operation to trigger the export
_ = export_op.numpy()

print("ONNX model exported to gemma3.onnx")
```