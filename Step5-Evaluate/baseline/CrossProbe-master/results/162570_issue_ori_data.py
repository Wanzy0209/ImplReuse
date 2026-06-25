```python
import tensorflow as tf

# Conversion: torch.nn.Module -> tf.Module
class Net(tf.Module):
    # Conversion: forward -> __call__
    # Added @tf.function to ensure tracing similar to ONNX export
    @tf.function
    def __call__(self, x, y):
        # Conversion: torch.atan2 -> tf.math.atan2
        return tf.math.atan2(x, y)


net = Net()
# Conversion: torch.tensor -> tf.constant
x = tf.constant([0.0])
y = tf.constant([0.0])

# Conversion: torch.onnx.export -> tf.saved_model.save
# Note: SavedModel saves to a directory, not a single file
tf.saved_model.save(net, "atan2_saved_model")

# Conversion: ort.InferenceSession -> tf.saved_model.load
loaded_net = tf.saved_model.load("atan2_saved_model")

# Run original model
tf_result = net(x, y)

# Run loaded model
# Note: In TF, we call the loaded object directly.
# The original code used specific input/output names, TF handles this via signatures,
# but direct call is the standard equivalent.
loaded_result = loaded_net(x, y)

print("tf_result", tf_result)
print("loaded_result", loaded_result)
```