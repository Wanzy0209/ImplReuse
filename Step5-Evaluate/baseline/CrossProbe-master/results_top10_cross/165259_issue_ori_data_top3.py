```python
import tensorflow as tf

# Conversion: torch.nn.Module -> tf.keras.Model
class PoseWrapper(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    # Conversion: forward -> call
    def call(self, x):
        output = self.model(x)
        return output["pose"]

model = PoseWrapper(original_model)

# Conversion: torch.ones -> tf.ones
# Note: TensorFlow uses tuples/lists for shape definition
input = tf.ones((1, 1, 512, 512))

# Conversion: torch.onnx.export -> model.save
# Note: TensorFlow models are dynamic by default, handling variable batch sizes without explicit arguments.
# The context provided maps to saving the model (architecture + weights) to HDF5.
# input_names/output_names are ONNX specific and not directly applicable to standard TF saving.
model.save(output_path)
```