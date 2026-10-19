```python
import tensorflow as tf
from transformers import TFAutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
# Conversion: Use TFAutoModelForCausalLM for TensorFlow
model = TFAutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it")


messages = [
    {"role": "user", "content": "Who are you?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="tf",  # Conversion: Changed "pt" to "tf" for TensorFlow tensors
)
# Conversion: TensorFlow handles device placement automatically (e.g., on GPU if available)


example_inputs = (inputs["input_ids"], inputs["attention_mask"])

# Conversion: torch.export.export captures a static graph of the model.
# In TensorFlow, the standard equivalent for exporting a model is tf.saved_model.save.
# This saves the model to disk in the SavedModel format.
tf.saved_model.save(model, "exported_model")
```