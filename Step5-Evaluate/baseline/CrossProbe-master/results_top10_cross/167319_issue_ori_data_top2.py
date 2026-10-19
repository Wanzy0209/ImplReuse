```python
import tensorflow as tf
from tensorflow.keras.optimizers import AdamW
from omegaconf import OmegaConf

# Create a config with nested structure
cfg = OmegaConf.create({
    'model': {'betas': [0.9, 0.999]},
    'data': {'batch_size': 32},
    # ... other configurations
})

# Create optimizer with OmegaConf ListConfig (no error raised)
# Conversion: torch.nn.Linear -> tf.keras.layers.Dense
model = tf.keras.layers.Dense(1, input_shape=(10,))

# Conversion: torch.optim.AdamW -> tf.keras.optimizers.AdamW
# Note: TF optimizers use beta_1 and beta_2 instead of a betas tuple.
# TF optimizers do not take model.parameters() at initialization.
optimizer = AdamW(learning_rate=1e-3, beta_1=cfg.model.betas[0], beta_2=cfg.model.betas[1])

# Save checkpoint
# Conversion: torch.save -> tf.train.Checkpoint
# Note: TF Checkpoint objects manage the state directly.
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
checkpoint.save('checkpoint')

# The entire config tree is serialized due to ListConfig._parent reference!
# This causes:
# 1. Unexpectedly large checkpoint files
# 2. PyTorch 2.6+ loading failures with weights_only=True
# Conversion Note: The serialization issue described above is specific to PyTorch's torch.save.
# TensorFlow's tf.train.Checkpoint saves tensor values and does not pickle the OmegaConf object.
```