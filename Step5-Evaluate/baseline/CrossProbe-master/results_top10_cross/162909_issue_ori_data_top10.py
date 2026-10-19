```python
from transformers import TFAutoModelForCausalLM, AutoTokenizer
import tensorflow as tf

# Conversion: torch.xpu.is_available checks for Intel XPU.
# In TensorFlow, we check for GPU availability.
# Note: TensorFlow does not have a direct public 'xpu' check in the main API like PyTorch.
print("Using device:", "gpu" if len(tf.config.list_physical_devices('GPU')) > 0 else "cpu")

model_name = "./Qwen3-06B"  # path to local model

# Conversion: Using TFAutoModelForCausalLM for TensorFlow.
# device_map="auto" is specific to PyTorch/accelerate. TF handles device placement automatically or via context managers.
# torch_dtype is PyTorch specific. For TF, use tf.keras.mixed_precision or load in float16 if supported by the model.
model = TFAutoModelForCausalLM.from_pretrained(
    model_name
    # device_map="auto",  # Not supported in TF Transformers
    # torch_dtype=torch.float16 # Not supported in TF
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```