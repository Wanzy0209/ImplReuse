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

# Conversion: PyTorch AdamW 'betas' argument maps to TF 'beta_1' and 'beta_2'
optimizer = AdamW(learning_rate=1e-3, beta_1=cfg.model.betas[0], beta_2=cfg.model.betas[1])

# Save checkpoint
# Conversion: torch.save -> tf.train.Checkpoint
# Note: TF Checkpoints save variable values, not the Python object graph.
# Therefore, the OmegaConf serialization issue mentioned below is specific to PyTorch's pickle mechanism.
checkpoint = tf.train.Checkpoint(optimizer=optimizer)
checkpoint.save('checkpoint')

# The entire config tree is serialized due to ListConfig._parent reference!
# This causes:
# 1. Unexpectedly large checkpoint files
# 2. PyTorch 2.6+ loading failures with weights_only=True
```