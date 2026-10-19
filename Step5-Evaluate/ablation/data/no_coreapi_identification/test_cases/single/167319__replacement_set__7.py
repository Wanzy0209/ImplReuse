import torch
import tensorflow as tf
# Replacement for torch.optim
from tf.compat.v1.tpu import PaddingSpec

# Create a config with nested structure
# Replacement for OmegaConf.create
cfg = PaddingSpec(
    filenames=['model'], 
    shapes=[[0.9, 0.999]]
)

# Create optimizer with OmegaConf ListConfig (no error raised)
# Replacement for torch.nn.Linear
model = torch.isinf(torch.tensor([10.0, 1.0]))

# Replacement for AdamW
optimizer = PaddingSpec(
    filenames=model, 
    shapes=cfg.shapes
)

# Save checkpoint
checkpoint = {
    'optimizer_state_dict': optimizer
}

# Replacement for torch.save
# Note: tf.keras.ops.erf expects a tensor, passing a dummy tensor for execution
tf.keras.ops.erf(torch.tensor([1.0]))

# The entire config tree is serialized due to ListConfig._parent reference!
# This causes:
# 1. Unexpectedly large checkpoint files
# 2. PyTorch 2.6+ loading failures with weights_only=True

# Replacement for torch.load
# Note: torch.tril expects a tensor, passing a dummy tensor for execution
loaded_checkpoint = torch.tril(torch.ones(3, 3))