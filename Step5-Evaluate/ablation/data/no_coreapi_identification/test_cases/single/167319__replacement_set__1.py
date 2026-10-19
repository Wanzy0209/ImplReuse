import torch
import tensorflow as tf

# Create a config with nested structure
# OmegaConf.create -> tf.compat.v1.Summary.Value
cfg = tf.compat.v1.Summary.Value(tag="config", simple_value=0)

# Create optimizer with OmegaConf ListConfig (no error raised)
# torch.nn.Linear -> torch.cat
# To make torch.cat runnable, we provide a list of tensors
dummy_tensor = torch.randn(10, 1)
model = torch.cat([dummy_tensor], dim=0)

# AdamW -> tf.compat.v1.Summary.Value
# torch.optim -> tf.compat.v1.Summary.Value (handled by import/usage)
# We pass the cfg object to mimic the original logic of passing the config
# Note: Summary.Value doesn't accept 'betas', so we adapt arguments to be valid
optimizer = tf.compat.v1.Summary.Value(tag="optimizer", simple_value=1e-3)

# Save checkpoint
checkpoint = {
    'optimizer_state_dict': optimizer # Value objects don't have state_dict, so we pass the object
}
# torch.save -> model.to
# We adapt the call to be valid for model.to (moving to device)
model.to(torch.device('cpu'))