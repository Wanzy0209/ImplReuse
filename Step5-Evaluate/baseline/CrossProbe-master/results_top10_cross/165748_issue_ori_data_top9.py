```python
import tensorflow as tf
import onnx
from tf2onnx import convert  # Conversion comment: PyTorch uses torch.onnx.export, TF uses tf2onnx

class SumModule(tf.keras.Model):
    def call(self, x):
        # Conversion comment: torch.sum(x, dim=1) -> tf.reduce_sum(x, axis=1)
        return tf.reduce_sum(x, axis=1)

# Conversion comment: torch.onnx.export equivalent. 
# Note: tf2onnx is an external library required for ONNX export from TensorFlow.
# dynamic_axes is handled by None in the TensorSpec shape.
convert.from_keras(
    SumModule(),
    input_signature=[tf.TensorSpec(shape=(None, 2), dtype=tf.float32, name="x")],
    output_path="onnx.pb",
    output_names=["sum"],
    opset=14
)

onnx_model = onnx.load("onnx.pb")
print(onnx_model.graph.input)
print(onnx_model.graph.output)
```