```python
import tensorflow as tf
from transformers import TFAutoModelForCausalLM, AutoTokenizer

# Conversion: torch.xpu.is_available checks for XPU hardware.
# In TensorFlow, we check for TPU availability using list_physical_devices.
# Note: The provided context suggested 'under_tpu_inference_context', but that checks
# a specific scope state rather than hardware availability.
print("Using device:", "tpu" if len(tf.config.list_physical_devices('TPU')) > 0 else "cpu")

model_name = "./Qwen3-06B"  # path to local model

# Conversion: torch_dtype=torch.float16 -> TF mixed precision policy
# Conversion: device_map="auto" is not directly supported in TF Transformers,
# placement is handled by the distribution strategy or default device placement.
tf.keras.mixed_precision.set_global_policy('mixed_float16')

model = TFAutoModelForCausalLM.from_pretrained(
    model_name
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```