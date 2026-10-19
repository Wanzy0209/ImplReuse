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
# Conversion: Define a Keras model equivalent to torch.nn.Linear
model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(10,))])

# Conversion: TF AdamW uses beta_1 and beta_2 arguments instead of a betas list.
# We unpack the values from the OmegaConf list.
optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, beta_1=cfg.model.betas[0], beta_2=cfg.model.betas[1])

# Save checkpoint
# Conversion: TF uses tf.train.Checkpoint to manage state, rather than a manual dictionary.
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
checkpoint.save('checkpoint')

# The entire config tree is serialized due to ListConfig._parent reference!
# This causes:
# 1. Unexpectedly large checkpoint files
# 2. PyTorch 2.6+ loading failures with weights_only=True
# Conversion Note: TF checkpoints use a binary protobuf format that stores tensor values,
# not the Python object graph. Therefore, the OmegaConf parent reference issue
# specific to PyTorch's pickle-based torch.save does not occur here.
```