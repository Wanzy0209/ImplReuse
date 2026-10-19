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
# Note: TensorFlow's standard export format is SavedModel.
# It natively supports dynamic shapes and saves parameters (weights) by default.
# Specific input/output names can be set via signatures if strict naming is required.
tf.saved_model.save(model, output_path)
```