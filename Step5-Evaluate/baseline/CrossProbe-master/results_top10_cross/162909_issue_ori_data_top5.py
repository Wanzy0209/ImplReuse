```python
from transformers import TFAutoModelForCausalLM, AutoTokenizer
import tensorflow as tf

# Conversion: Check for XPU availability. 
# In TensorFlow, we check for physical devices. Note: XPU support requires specific builds (e.g., intel_extension_for_tensorflow).
# The provided context API (is_ubsan_enabled) is a test utility and not applicable for hardware checks.
xpu_available = len(tf.config.list_physical_devices('XPU')) > 0
print("Using device:", "xpu" if xpu_available else "cpu")

model_name = "./Qwen3-06B"  # path to local model

# Conversion: Use TFAutoModelForCausalLM for TensorFlow.
# device_map and torch_dtype are PyTorch specific. 
# TensorFlow handles device placement automatically or via tf.distribute strategies.
model = TFAutoModelForCausalLM.from_pretrained(
    model_name
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```