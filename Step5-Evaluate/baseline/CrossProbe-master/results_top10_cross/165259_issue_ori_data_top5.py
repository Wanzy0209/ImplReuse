```python
import tensorflow as tf

# Conversion: torch.nn.Module -> tf.keras.Model
class PoseWrapper(tf.keras.Model):
    def __init__(self, model):
        super(PoseWrapper, self).__init__()
        self.model = model

    # Conversion: forward -> call
    def call(self, x):
        output = self.model(x)
        return output["pose"]

model = PoseWrapper(original_model)

# Conversion: torch.ones -> tf.ones
# Note: TensorFlow uses tuples for shape definition
input = tf.ones((1, 1, 512, 512))

# Conversion: torch.onnx.export -> tf.saved_model.save
# Note: TensorFlow uses the SavedModel format for exporting.
# Dynamic shapes are handled natively by the TensorFlow graph.
# input_names and output_names are handled by the model's signature.
tf.saved_model.save(
    model,
    output_path,
)
```