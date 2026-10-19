```python
import tensorflow as tf
from diffusers import TFStableDiffusionPipeline, TFPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"

# Conversion: Enable mixed precision to match torch.float16 behavior
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Conversion: Use TFStableDiffusionPipeline for TensorFlow backend
pipe = TFStableDiffusionPipeline.from_pretrained(model_id)

# Conversion: Use TFPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe.scheduler = TFPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Conversion: TensorFlow automatically uses GPU if available, no explicit .to("cuda") needed
prompt = "a photo of an astronaut riding a horse on mars"
import time
start = time.time()
for i in range(10):
  image = pipe(prompt).images[0]
print("Time taken: ", time.time() - start)

image.save("astronaut_rides_horse.png")
```