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

# Create model (Linear layer equivalent)
# Conversion: torch.nn.Linear(10, 1) -> tf.keras.layers.Dense
model = tf.keras.layers.Dense(1, input_shape=(10,))

# Create optimizer with OmegaConf ListConfig (no error raised)
# Conversion: PyTorch AdamW takes 'betas' as a list. 
# TensorFlow AdamW takes 'beta_1' and 'beta_2' as separate arguments.
optimizer = AdamW(learning_rate=1e-3, beta_1=cfg.model.betas[0], beta_2=cfg.model.betas[1])

# Save checkpoint
# Conversion: PyTorch uses state_dict() and torch.save().
# TensorFlow uses tf.train.Checkpoint to manage and save optimizer state.
checkpoint = tf.train.Checkpoint(optimizer=optimizer)
checkpoint.save('checkpoint')

# The entire config tree is serialized due to ListConfig._parent reference!
# This causes:
# 1. Unexpectedly large checkpoint files
# 2. PyTorch 2.6+ loading failures with weights_only=True
# Note: In TensorFlow, arguments are passed by value, so the optimizer 
# does not retain a reference to the OmegaConf object, preventing this issue.
```