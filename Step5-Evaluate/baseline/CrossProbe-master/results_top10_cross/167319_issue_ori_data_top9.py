```python
import tensorflow as tf
from tensorflow.keras.optimizers import AdamW
from omegaconf import OmegaConf
import pickle

# Create a config with nested structure
cfg = OmegaConf.create({
    'model': {'betas': [0.9, 0.999]},
    'data': {'batch_size': 32},
    # ... other configurations
})

# Create optimizer with OmegaConf ListConfig (no error raised)
# Conversion: tf.keras.layers.Dense is the TF equivalent of nn.Linear
model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(10,))])

# Conversion: TF AdamW uses beta_1 and beta_2 instead of betas list
# Conversion: TF optimizers are initialized separately and built with model variables
optimizer = AdamW(learning_rate=1e-3, beta_1=cfg.model.betas[0], beta_2=cfg.model.betas[1])
optimizer.build(model.trainable_variables)

# Save checkpoint
# Conversion: optimizer.get_weights() retrieves the state (similar to state_dict())
checkpoint = {
    'optimizer_weights': optimizer.get_weights()
}

# Conversion: Use pickle to save the dictionary, mimicking torch.save
with open('checkpoint.pkl', 'wb') as f:
    pickle.dump(checkpoint, f)

# The entire config tree is serialized due to ListConfig._parent reference!
# This causes:
# 1. Unexpectedly large checkpoint files
# 2. PyTorch 2.6+ loading failures with weights_only=True
```