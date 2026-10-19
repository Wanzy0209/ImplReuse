```python
import tensorflow as tf
# Conversion: Use TensorFlow-specific pipeline and scheduler classes
from diffusers import TFStableDiffusionPipeline, TFDPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
# Conversion: Enable mixed precision policy to match torch_dtype=torch.float16 behavior
tf.keras.mixed_precision.set_global_policy('mixed_float16')
pipe = TFStableDiffusionPipeline.from_pretrained(model_id)
# Conversion: Use TF-specific scheduler configuration
pipe.scheduler = TFDPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# Conversion: TensorFlow handles device placement automatically; explicit .to("cuda") is not used
# pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
import time
start = time.time()
for i in range(10):
  image = pipe(prompt).images[0]
print("Time taken: ", time.time() - start)

image.save("astronaut_rides_horse.png")
```