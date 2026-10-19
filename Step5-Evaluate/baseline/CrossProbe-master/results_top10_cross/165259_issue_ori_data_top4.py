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
# Note: PyTorch default dtype is float32, explicit in TF for safety
input = tf.ones((1, 1, 512, 512), dtype=tf.float32)

# Conversion: torch.onnx.export -> tf.saved_model.save
# To handle input_names, output_names, and dynamic_shapes, we use a tf.function signature.
# dynamic_shapes=({0: "batch"},) maps to None in the TensorSpec shape.
# input_names=["input"] maps to name="input" in TensorSpec.
# output_names=["pose"] maps to the key in the return dictionary.

@tf.function(input_signature=[tf.TensorSpec(shape=[None, 1, 512, 512], dtype=tf.float32, name="input")])
def export_fn(x):
    return {"pose": model(x)}

tf.saved_model.save(model, output_path, signatures=export_fn)
```