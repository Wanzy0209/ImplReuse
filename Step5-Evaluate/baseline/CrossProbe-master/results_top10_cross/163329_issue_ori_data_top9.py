```python
import tensorflow as tf
# Conversion: torch._logging.set_logs(recompiles=True) -> tf.get_logger().setLevel
# Note: TF does not have a direct 'recompiles' log flag. Setting to INFO for visibility.
tf.get_logger().setLevel('INFO')

# Conversion: diffusers.FluxPipeline -> keras_cv.models.StableDiffusion
# Note: FluxPipeline is PyTorch-specific. Using KerasCV Stable Diffusion as the closest TF equivalent.
from keras_cv.models import StableDiffusion

# Conversion: torch_dtype=torch.bfloat16 -> mixed precision policy
tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

# Conversion: .to('cuda') -> TF automatically uses GPU if available
# Conversion: compile_repeated_blocks -> handled by jit_compile=True
pipe = StableDiffusion(img_height=1024, img_width=1024, jit_compile=True)

prompt = "A cat holding a sign that says hello world"

# Conversion: torch.Generator("cpu").manual_seed(0) -> tf.random.Generator.from_seed(0)
# Note: KerasCV text_to_image accepts a seed integer directly.
rng = tf.random.Generator.from_seed(0)

# Conversion: pipe(...) -> pipe.text_to_image(...)
# Note: Mapping arguments: num_inference_steps->num_steps, guidance_scale->cfg_scale
image = pipe.text_to_image(
    prompt,
    batch_size=1,
    num_steps=50,
    cfg_scale=3.5,
    # max_sequence_length is not a parameter in KerasCV SD simple API
    seed=0
)

# Conversion: .images[0] -> image[0] (TensorFlow returns a tensor array)
# Conversion: .save() -> PIL Image save
from PIL import Image
Image.fromarray(image[0].astype("uint8")).save("flux-dev.png")
```