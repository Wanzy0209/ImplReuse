# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
from torch import optim
import matplotlib.pyplot as plt

# 1. Use a tensor learning rate. The choice of optimizer doesn't matter.
lr = 1.0
x = torch.tensor(0.0, device='meta')
opt = optim.AdamW([x], lr=torch.tensor(lr))

# 2. Initialize our chained schedulers.
milestone, total_steps = 40, 100
start_factor, end_factor = 0.2, 1.0
warmup = optim.lr_scheduler.LinearLR(opt, start_factor, end_factor, total_iters=milestone)
decay = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps-milestone)

# Note that each scheduler's base_lr aliases with the optimizer's initial_lr.
assert warmup.base_lrs[0].is_set_to(opt.param_groups[0]['initial_lr'])
assert decay.base_lrs[0].is_set_to(opt.param_groups[0]['initial_lr'])

# 3. Initialize our SequentialLR.
scheduler = optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, decay], milestones=[milestone])

# SequentialLR.__init__ aliases our optimizer's lr and initial_lr.
# for group in self.optimizer.param_groups:
#     group["lr"] = group["initial_lr"]
assert opt.param_groups[0]['lr'].is_set_to(opt.param_groups[0]['initial_lr'])

# Which means they're also aliased with the base_lrs of each scheduler!
assert (warmup.base_lrs[0].data_ptr() 
    == decay.base_lrs[0].data_ptr()
    == opt.param_groups[0]['initial_lr'].data_ptr() 
    == opt.param_groups[0]['lr'].data_ptr())

# It also calls _initial_step() on our LinearLR, setting all tensors to start_factor * lr.
assert (start_factor * lr
    == opt.param_groups[0]['lr']
    == opt.param_groups[0]['initial_lr']
    == warmup.base_lrs[0]
    == decay.base_lrs[0])

# 4. Now if we do something which breaks the aliases, e.g. save/load the optimizer's state_dict...
opt.load_state_dict(opt.state_dict())
assert not warmup.base_lrs[0].data_ptr() == opt.param_groups[0]['initial_lr'].data_ptr()

# Our base_lrs are incorrect. The LinearLR will work as intended (since get_lr updates based on the
# previous iteration, not base_lr), but the outputs of further schedulers are incorrectly scaled by 
# start_factor. In our case, these are the outputs of CosineAnnealingLR which appear after the milestone.
lrs = []
for step in range(total_steps):
    scheduler.step()
    lrs.append(scheduler.get_last_lr()[0].item())

plt.plot(lrs)
plt.show()