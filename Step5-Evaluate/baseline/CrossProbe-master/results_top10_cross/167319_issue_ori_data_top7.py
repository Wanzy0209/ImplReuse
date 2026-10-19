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
# Conversion: torch.nn.Linear -> tf.keras.layers.Dense
# We wrap the layer in a Sequential model to mimic the PyTorch Module structure
model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(10,))])

# Conversion: torch.optim.AdamW -> tf.keras.optimizers.AdamW
# Note: TF AdamW uses beta_1 and beta_2 arguments instead of a betas list
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=1e-3,
    beta_1=cfg.model.betas[0],
    beta_2=cfg.model.betas[1]
)

# Save checkpoint
# Conversion: torch.save -> tf.train.Checkpoint
# Note: tf.train.Checkpoint saves the state of trackable objects (variables).
# It does not pickle the entire object tree like torch.save, so the OmegaConf
# reference issue described below does not apply to TF checkpoints.
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
checkpoint.save('checkpoint') # Saves to checkpoint-1.data-00000-of-00001, etc.

# The entire config tree is serialized due to ListConfig._parent reference!
# This causes:
# 1. Unexpectedly large checkpoint files
# 2. PyTorch 2.6+ loading failures with weights_only=True
# (Note: This specific serialization issue is specific to PyTorch's pickle-based saving)
```