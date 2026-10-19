```python
import tensorflow as tf
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
# Conversion: Use from_pt=True to load PyTorch weights in TensorFlow
pipe = StableDiffusionPipeline.from_pretrained(model_id, from_pt=True)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# Conversion: TensorFlow handles device placement automatically
# pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
import time
start = time.time()
for i in range(10):
  image = pipe(prompt).images[0]
print("Time taken: ", time.time() - start)

image.save("astronaut_rides_horse.png")
```