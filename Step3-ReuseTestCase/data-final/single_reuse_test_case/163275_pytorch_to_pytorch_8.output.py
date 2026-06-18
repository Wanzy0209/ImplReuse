import torch
from torch.optim.swa_utils import get_ema_multi_avg_fn

# Setup data compatible with the similar API (lists of tensors)
# Using float16 to match the dtype context of the original bug report
ema_params = [torch.rand((1024, 1024), device="cuda", dtype=torch.float16)]
model_params = [torch.rand((1024, 1024), device="cuda", dtype=torch.float16)]

# Instantiate the similar API
# The original bug was about handling arguments, so we pass a specific decay value
ema_update_fn = get_ema_multi_avg_fn(decay=0.9)

@torch.compile
def run_ema_update(ema, model):
    # Adapted call site: using the similar API
    # The third argument is typically for buffers, passing None as in the source
    return ema_update_fn(ema, model, None)

run_ema_update(ema_params, model_params)