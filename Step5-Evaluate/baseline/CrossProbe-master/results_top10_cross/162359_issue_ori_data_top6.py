```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define custom schedules to mimic PyTorch behavior
class LinearLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, start_factor, end_factor, total_iters):
        self.initial_lr = initial_lr
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        total_iters = tf.cast(self.total_iters, tf.float32)
        progress = step / total_iters
        # Linear interpolation from start_factor to end_factor
        return self.initial_lr * (self.start_factor + (self.end_factor - self.start_factor) * progress)

class CosineAnnealingLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, T_max, eta_min=0.0):
        self.initial_lr = initial_lr
        self.T_max = T_max
        self.eta_min = eta_min

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        T_max = tf.cast(self.T_max, tf.float32)
        # Cosine annealing formula
        return self.eta_min + (self.initial_lr - self.eta_min) * (1 + tf.cos(np.pi * step / T_max)) / 2

class SequentialLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, schedulers, milestones):
        self.schedulers = schedulers
        self.milestones = milestones

    def __call__(self, step):
        # Conversion Note: TF schedules are functional. We use tf.cond to switch between schedules.
        return tf.cond(
            step < self.milestones[0],
            lambda: self.schedulers[0](step),
            lambda: self.schedulers[1](step - self.milestones[0])
        )

# 1. Use a tensor learning rate. The choice of optimizer doesn't matter.
lr = 1.0
# Conversion Note: TF does not have a 'meta' device. We use a standard variable.
x = tf.Variable(0.0)
# Conversion Note: TF optimizers take a float or a schedule. We initialize with float lr.
opt = tf.keras.optimizers.AdamW(learning_rate=lr)

# 2. Initialize our chained schedulers.
milestone, total_steps = 40, 100
start_factor, end_factor = 0.2, 1.0
# Conversion Note: We pass the initial lr value directly to the schedule constructors.
warmup = LinearLR(lr, start_factor, end_factor, total_iters=milestone)
decay = CosineAnnealingLR(lr, T_max=total_steps-milestone)

# Note that each scheduler's base_lr aliases with the optimizer's initial_lr.
# Conversion Note: TF schedules are stateless functions and do not alias memory with optimizer parameters.
# assert warmup.base_lrs[0].is_set_to(opt.param_groups[0]['initial_lr'])
# assert decay.base_lrs[0].is_set_to(opt.param_groups[0]['initial_lr'])

# 3. Initialize our SequentialLR.
scheduler = SequentialLR(schedulers=[warmup, decay], milestones=[milestone])

# SequentialLR.__init__ aliases our optimizer's lr and initial_lr.
# Conversion Note: In TF, we assign the schedule instance to the optimizer's learning_rate.
opt.learning_rate = scheduler

# for group in self.optimizer.param_groups:
#     group["lr"] = group["initial_lr"]
# Conversion Note: TF does not use param_groups with mutable lr dicts in this way.
# assert opt.param_groups[0]['lr'].is_set_to(opt.param_groups[0]['initial_lr'])

# Which means they're also aliased with the base_lrs of each scheduler!
# Conversion Note: TF schedules calculate values on the fly; they do not share memory pointers.
# assert (warmup.base_lrs[0].data_ptr() 
#     == decay.base_lrs[0].data_ptr()
#     == opt.param_groups[0]['initial_lr'].data_ptr() 
#     == opt.param_groups[0]['lr'].data_ptr())

# It also calls _initial_step() on our LinearLR, setting all tensors to start_factor * lr.
# Conversion Note: We verify the value at step 0 instead of checking memory aliases.
initial_val = scheduler(0).numpy()
assert abs(start_factor * lr - initial_val) < 1e-6
# assert (start_factor * lr
#     == opt.param_groups[0]['lr']
#     == opt.param_groups[0]['initial_lr']
#     == warmup.base_lrs[0]
#     == decay.base_lrs[0])

# 4. Now if we do something which breaks the aliases, e.g. save/load the optimizer's state_dict...
# Conversion Note: TF uses get_weights/set_weights. This does not break the schedule object reference.
opt.set_weights(opt.get_weights())
# assert not warmup.base_lrs[0].data_ptr() == opt.param_groups[0]['initial_lr'].data_ptr()

# Our base_lrs are incorrect. The LinearLR will work as intended (since get_lr updates based on the
# previous iteration, not base_lr), but the outputs of further schedulers are incorrectly scaled by 
# start_factor. In our case, these are the outputs of CosineAnnealingLR which appear after the milestone.
# Conversion Note: In TensorFlow, the schedule logic remains correct after loading weights because 
# the schedule is a separate object. The bug described in the PyTorch comment does not occur here.

lrs = []
for step in range(total_steps):
    # Conversion Note: TF schedules are called with the current step. 
    # We do not call scheduler.step() to update internal state.
    lrs.append(scheduler(step).numpy())

plt.plot(lrs)
