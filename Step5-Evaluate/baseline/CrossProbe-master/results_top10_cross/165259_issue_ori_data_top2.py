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
# Note: Renamed 'input' to 'input_data' to avoid shadowing Python built-in
input_data = tf.ones((1, 1, 512, 512))

# Conversion: torch.onnx.export -> tf.saved_model.save
# Note: TensorFlow uses the SavedModel format for exporting. 
# Dynamic shapes and input/output names are handled via signatures.
tf.saved_model.save(model, output_path)
```