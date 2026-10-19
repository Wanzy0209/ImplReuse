```python
from transformers import TFAutoModelForCausalLM, AutoTokenizer
import tensorflow as tf

# Conversion: torch.xpu.is_available -> tf.config.list_physical_devices('GPU')
# Note: TensorFlow does not have a specific 'xpu' check in the main API, checking for GPU instead.
print("Using device:", "gpu" if tf.config.list_physical_devices('GPU') else "cpu")

model_name = "./Qwen3-06B"  # path to local model

# Conversion: AutoModelForCausalLM -> TFAutoModelForCausalLM
# Conversion: device_map is not supported in TF version; TF handles device placement automatically.
# Conversion: torch_dtype is not supported; TF uses tf.float16 or mixed precision policies.
model = TFAutoModelForCausalLM.from_pretrained(
    model_name,
    from_pt=True  # Load from PyTorch weights if necessary
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```