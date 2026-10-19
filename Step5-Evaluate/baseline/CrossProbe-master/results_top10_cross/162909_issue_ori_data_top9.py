```python
from transformers import TFAutoModelForCausalLM, AutoTokenizer
import tensorflow as tf

# Conversion: TensorFlow checks for physical GPUs instead of XPU
print("Using device:", "gpu" if tf.config.list_physical_devices('GPU') else "cpu")

model_name = "./Qwen3-06B"  # path to local model

# Conversion: Use TFAutoModelForCausalLM for TensorFlow
# Conversion: torch_dtype is not a valid argument for TF models; precision is handled via tf.keras.mixed_precision
model = TFAutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```