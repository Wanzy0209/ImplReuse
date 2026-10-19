```python
import tensorflow as tf
# Conversion: Switched to TensorFlow-specific pipeline and scheduler classes
from diffusers import TFStableDiffusionPipeline, TFDPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"

# Use the TFDPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
# Conversion: torch_dtype is handled via TensorFlow's mixed precision policy or defaults
pipe = TFStableDiffusionPipeline.from_pretrained(model_id)
pipe.scheduler = TFDPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# Conversion: TensorFlow handles device placement automatically; no explicit .to() method needed
# pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
import time
start = time.time()
for i in range(10):
  image = pipe(prompt).images[0]
print("Time taken: ", time.time() - start)

image.save("astronaut_rides_horse.png")
```