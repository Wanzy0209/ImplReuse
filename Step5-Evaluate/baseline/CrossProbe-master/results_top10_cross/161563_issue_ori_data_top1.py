```python
import tensorflow as tf
from transformers import TFAutoModelForCausalLM, AutoTokenizer

# Conversion: Use TFAutoModelForCausalLM for TensorFlow
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
model = TFAutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it")


messages = [
    {"role": "user", "content": "Who are you?"},
]

# Conversion: Use return_tensors="tf" for TensorFlow tensors
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="tf",
)
# Conversion: TensorFlow handles device placement automatically or via tf.device context, no .to() needed


example_inputs = (inputs["input_ids"], inputs["attention_mask"])

# Conversion: Use tf.saved_model.save to export the model graph and variables
# Note: The context-provided Target API (tf_record_test.testIterator) is unrelated to model exporting and is ignored.
tf.saved_model.save(model, "exported_gemma")
```