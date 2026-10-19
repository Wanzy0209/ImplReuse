```python
from transformers import TFAutoModelForCausalLM, AutoTokenizer
import tensorflow as tf

# Conversion: Replaced torch.xpu.is_available with TensorFlow device checking
# The provided context API was irrelevant to hardware availability checks.
print("Using device:", "xpu" if tf.config.list_physical_devices('GPU') else "cpu")

model_name = "./Qwen3-06B"  # path to local model

# Conversion: Switched to TFAutoModelForCausalLM and removed PyTorch-specific args (device_map, torch_dtype)
model = TFAutoModelForCausalLM.from_pretrained(
    model_name
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```