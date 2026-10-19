```python
from transformers import TFAutoModelForCausalLM, AutoTokenizer
import tensorflow as tf

# Conversion: torch.xpu.is_available checks for Intel XPU hardware.
# In TensorFlow, we check for GPU availability using tf.config.list_physical_devices.
# Note: The provided context API (is_msan_enabled) is for checking Memory Sanitizer
# build flags, not hardware availability, so the standard GPU check is used instead.
print("Using device:", "gpu" if len(tf.config.list_physical_devices('GPU')) > 0 else "cpu")

model_name = "./Qwen3-06B"  # path to local model

model = TFAutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # device_map is supported in TF versions of transformers
    # Conversion: torch.float16 -> tf.float16
    dtype=tf.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```