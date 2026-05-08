import io
import torch
from torch import optim, nn

opt = optim.SGD([nn.Parameter(torch.tensor(0.0))], lr=0.1)
scheduler = optim.swa_utils.SWALR(opt, swa_lr=0.01)

# This state dict includes a non-serializable anneal_func
bad_state_dict = scheduler.state_dict()

buffer = io.BytesIO()
torch.save(bad_state_dict, buffer)
buffer.seek(0)

torch.load(buffer, weights_only=True)

# UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint.
#         (1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
#         (2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
#         WeightsUnpickler error: Unsupported global: GLOBAL getattr was not an allowed global by default. Please use `torch.serialization.add_safe_globals([getattr])` or the `torch.serialization.safe_globals([getattr])` context manager to allowlist this global if you trust this class/function.