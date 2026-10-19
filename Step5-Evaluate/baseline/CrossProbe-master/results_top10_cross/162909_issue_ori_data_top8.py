```python
from transformers import TFAutoModelForCausalLM, AutoTokenizer
import tensorflow as tf

# Conversion: torch.xpu.is_available checks for Intel XPU.
# TensorFlow checks for hardware availability (e.g., GPU) via tf.config.list_physical_devices.
# Note: The suggested target API 'check_pydot' is for visualization dependencies and is not applicable for hardware checks.
gpus = tf.config.list_physical_devices('GPU')
print("Using device:", "gpu" if gpus else "cpu")

model_name = "./Qwen3-06B"  # path to local model

# Conversion: Switched to TFAutoModelForCausalLM for TensorFlow backend.
# device_map="auto" and torch_dtype are PyTorch-specific arguments.
# In TensorFlow, device placement is handled by the runtime, and dtype is often managed via policies or model loading.
model = TFAutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```