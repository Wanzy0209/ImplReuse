import torch
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

# Load the tokenizer and the TensorFlow version of the model
model_id = "google/gemma-3-270m-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = TFAutoModelForCausalLM.from_pretrained(model_id)

# Prepare input messages
messages = [
    {"role": "user", "content": "Who are you?"},
]

# Tokenize inputs, returning TensorFlow tensors
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="tf",
)

example_inputs = (inputs["input_ids"], inputs["attention_mask"])

# Define the computation function to be compiled
def model_computation(input_ids, attention_mask):
    return model(input_ids=input_ids, attention_mask=attention_mask)

# Attempt to compile using the similar API (tf.xla.experimental.compile)
# This corresponds to the torch.export.export call in the original bug report
try:
    compiled_result = tf.xla.experimental.compile(model_computation, example_inputs)
    print("XLA Compilation successful.")
    print("Result:", compiled_result)
except Exception as e:
    print(f"Error during XLA compilation: {e}")