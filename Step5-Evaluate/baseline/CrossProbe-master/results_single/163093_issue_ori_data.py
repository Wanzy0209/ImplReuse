# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
from torch import optim, nn

torch._logging.set_logs(recompiles=True)
device = "cuda"
model = nn.Linear(1, 1).to(device)

lr = torch.tensor(0.1)
opt = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1, min_lr=0.001)

# From https://docs.pytorch.org/tutorials/recipes/compiling_optimizer_lr_scheduler.html
@torch.compile(fullgraph=False)
def fn(metric):
    opt.step()
    scheduler.step(metric)

total_steps = 8
fake_metrics = torch.linspace(1.0, 0.0, total_steps, device=device)
fake_metrics[3:6] = fake_metrics[3]

for metric in fake_metrics:
    fn(metric)

# W0912 20:46:20.358000 89262 torch/_logging/_internal.py:1199] [0/0] Profiler function <class 'torch.autograd.profiler.record_function'> will be ignored
# V0912 20:46:20.564000 89262 torch/_dynamo/guards.py:4295] [2/1] [__recompiles] Recompiling function step in /workspace/pytorch/torch/optim/adam.py:213
# V0912 20:46:20.564000 89262 torch/_dynamo/guards.py:4295] [2/1] [__recompiles]     triggered by the following guard failure(s):
# V0912 20:46:20.564000 89262 torch/_dynamo/guards.py:4295] [2/1] [__recompiles]     - 2/0: expected type of 'self.param_groups[0]['lr']' to be a tensor type, ' but found <class 'float'>