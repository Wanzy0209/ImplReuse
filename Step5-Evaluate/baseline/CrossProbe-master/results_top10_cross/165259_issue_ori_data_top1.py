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
# Note: TensorFlow uses tuples/lists for shape arguments
input = tf.ones((1, 1, 512, 512))

# Conversion: torch.onnx.export -> tf.saved_model.save
# To map input_names, output_names, and dynamic_shapes, we define a tf.function signature.

# PyTorch: dynamic_shapes=({0: "batch"},) maps to None in the batch dimension
# PyTorch: input_names=["input"] maps to the name argument in TensorSpec
input_spec = tf.TensorSpec(shape=[None, 1, 512, 512], dtype=tf.float32, name="input")

# PyTorch: dynamo=True implies graph compilation. In TF, @tf.function traces the graph.
@tf.function(input_signature=[input_spec])
def serve(x):
    # PyTorch: output_names=["pose"] maps to the dictionary key returned
    return {"pose": model(x)}

# PyTorch: export_params=True is default behavior in TF (variables are saved)
# PyTorch: output_path is the destination directory
tf.saved_model.save(model, output_path, signatures=serve)
```