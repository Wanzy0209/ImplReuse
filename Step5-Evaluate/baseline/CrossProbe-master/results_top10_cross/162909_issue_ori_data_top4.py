```python
from transformers import TFAutoModelForCausalLM, AutoTokenizer
import tensorflow as tf

# Conversion: Check for GPU availability in TensorFlow.
# Note: The provided context API (is_tsan_enabled) is a test utility and not applicable for hardware detection.
print("Using device:", "xpu" if tf.config.list_physical_devices('GPU') else "cpu")

model_name = "./Qwen3-06B"  # path to local model

# Conversion: Use TFAutoModelForCausalLM for TensorFlow.
# Conversion: device_map is removed as it is specific to PyTorch/accelerate.
# Conversion: torch_dtype is removed; TF handles precision via tf.keras.mixed_precision or model loading.
# Conversion: Added from_pt=True assuming the local model contains PyTorch weights.
model = TFAutoModelForCausalLM.from_pretrained(
    model_name,
    from_pt=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```