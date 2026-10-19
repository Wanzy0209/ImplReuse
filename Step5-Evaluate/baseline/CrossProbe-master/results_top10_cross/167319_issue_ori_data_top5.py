```python
import tensorflow as tf
from omegaconf import OmegaConf

# Create a config with nested structure
cfg = OmegaConf.create({
    'model': {'betas': [0.9, 0.999]},
    'data': {'batch_size': 32},
    # ... other configurations
})

# Create optimizer with OmegaConf ListConfig (no error raised)
# Conversion Note: tf.keras.layers.Dense is the equivalent of torch.nn.Linear
model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(10,))])

# Conversion Note: TensorFlow's AdamW uses beta_1 and beta_2 arguments instead of a single betas list.
# We unpack the OmegaConf list here to match the functionality.
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=1e-3,
    beta_1=cfg.model.betas[0],
    beta_2=cfg.model.betas[1]
)

# Save checkpoint
# Conversion Note: TensorFlow uses tf.train.Checkpoint to manage state, rather than a raw dictionary.
checkpoint = tf.train.Checkpoint(optimizer=optimizer)
checkpoint.save('checkpoint')

# The entire config tree is serialized due to ListConfig._parent reference!
# This causes:
# 1. Unexpectedly large checkpoint files
# 2. PyTorch 2.6+ loading failures with weights_only=True
# Conversion Note: The issue described above is specific to PyTorch's pickle-based serialization.
# TensorFlow saves weights as binary tensors and does not serialize the configuration objects
# passed to the optimizer, so this specific issue does not manifest in TensorFlow.
```