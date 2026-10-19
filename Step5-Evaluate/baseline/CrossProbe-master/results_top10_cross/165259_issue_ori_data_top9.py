```python
import tensorflow as tf

class PoseWrapper(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, x):
        output = self.model(x)
        return output["pose"]

model = PoseWrapper(original_model)

# Conversion: torch.ones -> tf.ones
input = tf.ones((1, 1, 512, 512))

# Conversion: torch.onnx.export -> tf.saved_model.save
# To handle input_names, output_names, and dynamic_shapes, we define a concrete signature.
# dynamic_shapes=({0: "batch"},) maps to None in the batch dimension of TensorSpec.
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 1, 512, 512], dtype=tf.float32, name="input")])
def export_fn(x):
    # The wrapper returns the tensor, but we wrap it in a dict to match output_names=["pose"]
    return {"pose": model(x)}

tf.saved_model.save(
    model,
    output_path,
    signatures=export_fn
)
```