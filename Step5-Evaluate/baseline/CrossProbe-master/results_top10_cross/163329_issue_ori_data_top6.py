```python
import tensorflow as tf

# Conversion: torch._logging.set_logs(recompiles=True) -> Enable TF logging
tf.get_logger().setLevel('INFO')

from diffusers import FluxPipeline

# Conversion: torch_dtype=torch.bfloat16 -> dtype=tf.bfloat16
# Conversion: .to('cuda') -> .to('/GPU:0') (Keras 3 / TF device placement)
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", dtype=tf.bfloat16).to('/GPU:0')

# compile
# Conversion: compile_repeated_blocks() -> compile() (Standard TF/Keras model compilation)
pipe.transformer.compile()

prompt = "A cat holding a sign that says hello world"
# Conversion: torch.Generator("cpu").manual_seed(0) -> tf.random.Generator.from_seed(0)
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=tf.random.Generator.from_seed(0)
).images[0]
image.save("flux-dev.png")
```