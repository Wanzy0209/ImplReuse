```python
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
# Conversion: Device placement is handled automatically in TensorFlow
# .to(model.device) is not required

# Conversion: Prepare inputs as a tuple to match PyTorch structure
example_inputs = (inputs["input_ids"], inputs["attention_mask"])

# Conversion: torch.export.export captures the computation graph.
# In TensorFlow, we use tf.function to trace and create a graph.
@tf.function
def model_export(input_ids, attention_mask):
    return model(input_ids=input_ids, attention_mask=attention_mask)

# Get the concrete function (traced graph) using the example inputs
# This is the TensorFlow equivalent of the ExportedProgram
ep = model_export.get_concrete_function(*example_inputs)
```