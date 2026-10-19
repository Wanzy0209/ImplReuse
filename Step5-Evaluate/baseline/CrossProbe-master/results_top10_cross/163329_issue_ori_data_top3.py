```python
import tensorflow as tf
import keras_cv
from PIL import Image

# Conversion: torch._logging.set_logs(recompiles=True)
# TensorFlow uses the standard logging module or tf.get_logger()
tf.get_logger().setLevel('INFO')

# Conversion: from diffusers import FluxPipeline
# Note: FluxPipeline is specific to the PyTorch diffusers library. 
# There is no direct TensorFlow equivalent for the FLUX.1-dev model in standard libraries yet.
# We use KerasCV's StableDiffusionXL as the closest structural equivalent for a high-res diffusion pipeline.
from keras_cv.models import StableDiffusionXL

# Conversion: torch_dtype=torch.bfloat16 and .to('cuda')
# TensorFlow uses mixed precision policies and handles device placement automatically.
tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

# Conversion: pipe.transformer.compile_repeated_blocks()
# In TensorFlow, we enable XLA compilation (jit_compile) for performance optimization.
pipe = StableDiffusionXL(img_height=1024, img_width=1024, jit_compile=True)

prompt = "A cat holding a sign that says hello world"

# Conversion: generator=torch.Generator("cpu").manual_seed(0)
# KerasCV uses a seed integer for reproducibility.
# max_sequence_length is handled internally or is not exposed in the high-level API.
image = pipe.text_to_image(
    prompt,
    batch_size=1,
    num_steps=50,          # num_inference_steps
    guidance_scale=3.5,
    seed=0                 # generator
)[0]

# Conversion: image.save("flux-dev.png")
# KerasCV returns a numpy array, so we convert it to a PIL Image to save.
img_pil = Image.fromarray(image)
img_pil.save("flux-dev.png")
```