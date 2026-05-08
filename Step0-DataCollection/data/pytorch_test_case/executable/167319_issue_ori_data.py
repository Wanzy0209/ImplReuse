import torch
from torch.optim import AdamW
from omegaconf import OmegaConf

# Create a config with nested structure
cfg = OmegaConf.create({
    'model': {'betas': [0.9, 0.999]},
    'data': {'batch_size': 32},
    # ... other configurations
})

# Create optimizer with OmegaConf ListConfig (no error raised)
model = torch.nn.Linear(10, 1)
optimizer = AdamW(model.parameters(), lr=1e-3, betas=cfg.model.betas)

# Save checkpoint
checkpoint = {
    'optimizer_state_dict': optimizer.state_dict()
}
torch.save(checkpoint, 'checkpoint.pt')

# The entire config tree is serialized due to ListConfig._parent reference!
# This causes:
# 1. Unexpectedly large checkpoint files
# 2. PyTorch 2.6+ loading failures with weights_only=True