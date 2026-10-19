# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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


example_inputs = (inputs["input_ids"], inputs["attention_mask"])

# ONNX Export
torch.onnx.export(
    model,
    example_inputs,
    "gemma3.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    opset_version=17,
    do_constant_folding=True,
    dynamo = True,
)

print("ONNX model exported to gemma3.onnx")