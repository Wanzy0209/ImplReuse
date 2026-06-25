```python
import tensorflow as tf
# Conversion: Switched to TensorFlow-specific pipeline and scheduler classes
from diffusers import TFSstableDiffusionPipeline, TFDPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
# Conversion: Changed torch_dtype to dtype for TensorFlow
pipe = TFSstableDiffusionPipeline.from_pretrained(model_id, dtype=tf.float16)
# Conversion: Using TensorFlow-specific scheduler
pipe.scheduler = TFDPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# Conversion: TensorFlow handles device placement automatically if GPU is available; explicit .to() is not used
# pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
import time
start = time.time()
for i in range(10):
  image = pipe(prompt).images[0]
print("Time taken: ", time.time() - start)

image.save("astronaut_rides_horse.png")
```