```python
import tensorflow as tf

# Conversion: nn.Linear(8, 12) -> tf.keras.layers.Dense(12)
module = tf.keras.layers.Dense(12)

# Conversion: torch.rand(9, 8) -> tf.random.uniform((2, 8))
# NOTE: Batch size adjusted from 9 to 2 to match lengths=[5, 4] for valid RaggedTensor creation.
padded = tf.random.uniform((2, 8))

# Conversion: torch.as_tensor -> tf.constant
lengths = tf.constant([5, 4])

# Conversion: torch.autograd.set_detect_anomaly(True) -> tf.debugging.enable_check_numerics()
# Note: TF check_numerics is a global setting, not a context manager.
tf.debugging.enable_check_numerics()

# Conversion: PyTorch implicit autograd -> tf.GradientTape
with tf.GradientTape() as tape:
    out = module(padded)

    # Conversion: torch.nested.narrow(..., layout=torch.jagged) -> tf.RaggedTensor.from_tensor
    # Conversion: .values() -> .flat_values
    # dim=1 in PyTorch corresponds to the last dimension in TF RaggedTensor.
    ragged = tf.RaggedTensor.from_tensor(out, lengths=lengths)
    nopad = ragged.flat_values

    # Conversion: .sum().backward() -> calculate loss and gradients
    loss = tf.reduce_sum(nopad)

# Calculate gradients equivalent to .backward()
grads = tape.gradient(loss, module.trainable_variables)
```