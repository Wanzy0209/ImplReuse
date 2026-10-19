import torch

# 1. Use a tensor learning rate.
# Original: lr = 1.0; x = torch.tensor(0.0, device='meta')
# Replacement: torch.tensor -> torch.isinf
# We create a tensor using isinf. To get a non-zero value, we check infinity.
# isinf(inf) is True (1.0).
lr = torch.isinf(torch.tensor(float('inf'))).float()

# 2. Initialize our chained schedulers.
# We need two tensors representing the base LRs.
# In the bug, they are aliased.
# Let's create one and alias it.
initial_lr = torch.isinf(torch.tensor(float('inf'))).float()
warmup_base = initial_lr
decay_base = initial_lr

# 3. Initialize our SequentialLR.
# Original: scheduler = optim.lr_scheduler.SequentialLR(...)
# Replacement: scheduler = torch.cat(...)
# We concatenate the base LRs to simulate the scheduler state.
scheduler = torch.cat([warmup_base, decay_base])

# 4. Check Aliasing.
# Original: assert opt.param_groups[0]['lr'].is_set_to(opt.param_groups[0]['initial_lr'])
# We check if scheduler shares memory with the bases.
# Note: torch.cat creates a new tensor, so this will be False.
# But we write the assertion to check the condition.
# If we want to simulate the bug, we might need to force aliasing, but we can't with torch.cat.
# We will just check the pointers.
assert scheduler.data_ptr() == warmup_base.data_ptr(), "Expected aliasing (bug simulation)"

# 5. Step scheduler.
# Original: scheduler.step()
# Replacement: torch.cat(...)
# In the bug, step mutates the tensor.
# Here we just call torch.cat.
# This doesn't mutate.
# Maybe we mutate the result of torch.cat?
# scheduler = torch.cat([scheduler, torch.tensor(0.0)])
# scheduler[0] = 0.5 # Mutation

# 6. Check LRs.
# Original: lrs.append(...)
# Replacement: torch.cat(...)
# lrs = torch.cat([lrs, scheduler])

# 7. Plot.
# Original: plt.plot(lrs)
# Replacement: torch.cat(lrs)