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
# Conversion: torch.nn.Linear(10, 1) -> tf.keras.layers.Dense(1) wrapped in a Model
model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(10,))])

# Conversion: torch.optim.AdamW -> tf.keras.optimizers.AdamW
# Note: TF AdamW uses beta_1 and beta_2 arguments instead of a single betas list.
# We unpack the list from the config to match the TF API.
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=1e-3, 
    beta_1=cfg.model.betas[0], 
    beta_2=cfg.model.betas[1]
)

# Save checkpoint
# Conversion: torch.save -> tf.train.Checkpoint
# TF Checkpoint manages the state directly, we don't manually create a dict of state_dict.
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
checkpoint.save('checkpoint')

# The entire config tree is serialized due to ListConfig._parent reference!
# This causes:
# 1. Unexpectedly large checkpoint files
# 2. PyTorch 2.6+ loading failures with weights_only=True
# Conversion Note: This specific issue is PyTorch-specific. 
# TensorFlow Checkpoints save variable values and do not pickle the optimizer object graph,
# so the OmegaConf reference issue does not occur here.
```