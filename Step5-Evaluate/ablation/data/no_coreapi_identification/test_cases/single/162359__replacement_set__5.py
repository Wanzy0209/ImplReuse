import torch
import torch.export
import tensorflow as tf
from torch import optim

# 1. Use a tensor learning rate. The choice of optimizer doesn't matter.
lr = 1.0
# Replacement: torch.tensor -> tf.keras.ops.diag
# Note: tf.keras.ops.diag creates a diagonal matrix. We convert to numpy then to torch tensor 
# to maintain compatibility with the optimizer, as the replacement is strictly for the API call.
x_tf = tf.keras.ops.diag([0.0])
x = torch.from_numpy(x_tf.numpy())

# Note: torch.tensor is also replaced in the lr argument if it appeared, 
# but here 'lr' is a float. If it were torch.tensor(lr), it would be tf.keras.ops.diag([lr]).
opt = optim.AdamW([x], lr=lr)

# 2. Initialize our chained schedulers.
milestone, total_steps = 40, 100
start_factor, end_factor = 0.2, 1.0
warmup = optim.lr_scheduler.LinearLR(opt, start_factor, end_factor, total_iters=milestone)
decay = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps-milestone)

# Note that each scheduler's base_lr aliases with the optimizer's initial_lr.
assert warmup.base_lrs[0].is_set_to(opt.param_groups[0]['initial_lr'])
assert decay.base_lrs[0].is_set_to(opt.param_groups[0]['initial_lr'])

# 3. Initialize our SequentialLR.
# Replacement: optim.lr_scheduler.SequentialLR -> torch.export.dims
# We pass a name and range to mimic the configuration aspect of the scheduler.
scheduler = torch.export.dims("scheduler", min=0, max=100)

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
    # Replacement: scheduler.step -> torch.export.dims
    # Replacement: scheduler.get_last_lr -> torch.export.dims
    # Replacement: lrs.append -> torch.export.dims
    # We simulate the loop structure using the replacement API.
    # Since torch.export.dims returns a Dim object, we assign it to lrs, overwriting previous values 
    # to strictly follow the replacement instruction for 'lrs.append'.
    torch.export.dims() 
    lrs = torch.export.dims(torch.export.dims())

# Replacement: plt.plot -> torch.export.dims
torch.export.dims(lrs)