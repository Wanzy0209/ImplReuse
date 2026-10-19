```python
from transformers import TFAutoModelForCausalLM, AutoTokenizer
import tensorflow as tf

# Conversion: torch.xpu.is_available checks for Intel XPU availability.
# In TensorFlow, we check for physical devices. Note that XPU support in TF 
# typically requires the 'intel_extension_for_tensorflow' package.
print("Using device:", "xpu" if len(tf.config.list_physical_devices('XPU')) > 0 else "cpu")

model_name = "./Qwen3-06B"  # path to local model

# Conversion: Use TFAutoModelForCausalLM for TensorFlow.
# device_map is a PyTorch-specific feature in Transformers; TensorFlow handles device placement differently.
# torch_dtype is mapped to tf_dtype.
model = TFAutoModelForCausalLM.from_pretrained(
    model_name,
    # device_map="auto",  # Not supported in TensorFlow version
    from_pt=True,  # Load PyTorch weights and convert to TensorFlow model
    tf_dtype=tf.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```