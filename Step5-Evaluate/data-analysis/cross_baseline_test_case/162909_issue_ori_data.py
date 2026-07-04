```python
from transformers import TFAutoModelForCausalLM, AutoTokenizer
import tensorflow as tf

# Conversion: torch.xpu.is_available checks for Intel XPU.
# In TensorFlow, we check for physical GPUs. Note that Intel XPU support in TensorFlow
# is typically handled via the Intel Extension for TensorFlow (ITEX), where it appears as a GPU.
gpus = tf.config.list_physical_devices('GPU')
print("Using device:", "xpu" if gpus else "cpu")

model_name = "./Qwen3-06B"  # path to local model

# Conversion: TFAutoModelForCausalLM is the TensorFlow equivalent.
# device_map is not a parameter in TF; TF handles device placement automatically.
# from_pt=True is added to convert PyTorch weights to TensorFlow weights.
# torch_dtype=torch.float16 is handled in TF via mixed precision policies (e.g., tf.keras.mixed_precision.set_global_policy('mixed_float16')).
model = TFAutoModelForCausalLM.from_pretrained(
    model_name,
    from_pt=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```