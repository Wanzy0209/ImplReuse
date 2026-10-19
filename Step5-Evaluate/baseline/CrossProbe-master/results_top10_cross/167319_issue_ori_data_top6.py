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

# Conversion: torch.nn.Linear(10, 1) -> tf.keras.layers.Dense(1, input_shape=(10,))
# Wrapped in Sequential to provide a trainable variables interface
model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(10,))])

# Conversion: PyTorch AdamW uses 'betas' tuple, TensorFlow uses 'beta_1' and 'beta_2' floats.
# Create optimizer with OmegaConf ListConfig (no error raised)
optimizer = AdamW(learning_rate=1e-3, beta_1=cfg.model.betas[0], beta_2=cfg.model.betas[1])

# Conversion: PyTorch uses torch.save with a dictionary. TensorFlow uses tf.train.Checkpoint.
# Save checkpoint
checkpoint = tf.train.Checkpoint(optimizer=optimizer)
checkpoint.save('checkpoint')

# The entire config tree is serialized due to ListConfig._parent reference!
# This causes:
# 1. Unexpectedly large checkpoint files
# 2. PyTorch 2.6+ loading failures with weights_only=True
# Note: TensorFlow checkpoints use protobuf and save variable values, not the optimizer
# configuration objects, so this specific OmegaConf serialization issue does not occur.
```