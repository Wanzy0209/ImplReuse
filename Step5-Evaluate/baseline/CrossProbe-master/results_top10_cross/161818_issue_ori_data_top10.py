```python
import tensorflow as tf

# nn.Linear(8, 12) maps to a Dense layer with 12 units
module = tf.keras.layers.Dense(12, input_shape=(8,))

# torch.rand(9, 8) creates a random uniform tensor
padded = tf.random.uniform((9, 8))
# torch.as_tensor([5, 4]) creates a tensor
lengths = tf.constant([5, 4])

# torch.autograd.set_detect_anomaly(True) enables NaN/Inf checking
# TensorFlow uses a global debug flag for this
tf.debugging.enable_check_numerics()

# TensorFlow requires GradientTape to compute gradients (backward pass)
with tf.GradientTape() as tape:
    out = module(padded)

    # torch.nested.narrow creates a NestedTensor (jagged layout)
    # Conversion Note: The original code uses dim=1, but out.shape is (9, 12) and lengths sum to 9.
    # This implies dim=0 (batch dimension) is the intended logic for the split.
    # We use tf.RaggedTensor.from_row_lengths to mimic the nested structure.
    # .values() extracts the underlying buffer, equivalent to .flat_values in TF.
    rt = tf.RaggedTensor.from_row_lengths(out, lengths)
    nopad = rt.flat_values

    # nopad.sum().backward() computes the sum and gradients
    loss = tf.reduce_sum(nopad)

# Compute gradients
grads = tape.gradient(loss, module.trainable_variables)
```