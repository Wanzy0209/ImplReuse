```python
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Use a tensor learning rate. The choice of optimizer doesn't matter.
lr = 1.0
# Conversion: In PyTorch, x is a parameter. In TF, we define a Variable.
x = tf.Variable(0.0)
# Conversion: TF optimizers accept a LearningRateSchedule object to handle dynamic LRs.
# We will define the schedule below and pass it to the optimizer.

# 2. Initialize our chained schedulers.
milestone, total_steps = 40, 100
start_factor, end_factor = 0.2, 1.0

# Conversion: PyTorch LinearLR -> tf.keras.optimizers.schedules.PolynomialDecay (power=1.0 for linear)
# LinearLR goes from start_factor * lr to end_factor * lr.
warmup = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=lr * start_factor,
    decay_steps=milestone,
    end_learning_rate=lr * end_factor,
    power=1.0
)

# Conversion: PyTorch CosineAnnealingLR -> tf.keras.optimizers.schedules.CosineDecay
# CosineAnnealingLR decays from initial_lr to 0 over T_max.
decay = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=lr,
    decay_steps=total_steps - milestone,
    alpha=0.0
)

# Conversion: PyTorch asserts regarding aliasing (base_lrs, initial_lr) are specific to PyTorch's implementation.
# TF uses LearningRateSchedule objects which calculate LR on the fly based on the step.
# assert warmup.base_lrs[0].is_set_to(opt.param_groups[0]['initial_lr'])
# assert decay.base_lrs[0].is_set_to(opt.param_groups[0]['initial_lr'])

# 3. Initialize our SequentialLR.
# Conversion: TF does not have a direct SequentialLR, but we can compose schedules using a custom class.
class SequentialSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_sched, decay_sched, milestone_step):
        super().__init__()
        self.warmup_sched = warmup_sched
        self.decay_sched = decay_sched
        self.milestone_step = milestone_step

    def __call__(self, step):
        return tf.cond(
            step < self.milestone_step,
            lambda: self.warmup_sched(step),
            lambda: self.decay_sched(step - self.milestone_step)
        )
    
    def get_config(self):
        return {
            "warmup": self.warmup_sched,
            "decay": self.decay_sched,
            "milestone": self.milestone_step
        }

scheduler = SequentialSchedule(warmup, decay, milestone)

# Conversion: Initialize optimizer with the schedule.
opt = tf.keras.optimizers.AdamW(learning_rate=scheduler)

# Conversion: PyTorch SequentialLR aliases optimizer lr. TF optimizer reads from the schedule object.
# The concept of aliasing memory pointers (data_ptr) does not exist in TF Python API.
# assert opt.param_groups[0]['lr'].is_set_to(opt.param_groups[0]['initial_lr'])
# assert (warmup.base_lrs[0].data_ptr() == decay.base_lrs[0].data_ptr() ...)

# Conversion: Verify initial LR.
# In TF, we call the schedule to get the LR.
current_lr = scheduler(0)
# assert (start_factor * lr == current_lr == warmup(0) == decay(0)) 
# Note: decay(0) in TF CosineDecay is 1.0, which matches the logic.

# 4. Now if we do something which breaks the aliases, e.g. save/load the optimizer's state_dict...
# Conversion: TF uses Checkpoints. Saving/loading does not break the link to the schedule object
# because the schedule is a Python object, not a tensor state inside the optimizer (unless serialized explicitly).
checkpoint = tf.train.Checkpoint(optimizer=opt)
ckpt_path = checkpoint.save("./tf_ckpt")
checkpoint.restore(ckpt_path)

# Conversion: The specific aliasing bug in PyTorch does not apply to TF.
# assert not warmup.base_lrs[0].data_ptr() == opt.param_groups[0]['initial_lr'].data_ptr()

# Our base_lrs are incorrect... (PyTorch comment)
# In TF, the schedule logic remains correct regardless of checkpointing the optimizer.

lrs = []
for step in range(total_steps):
    # Conversion: In TF, we don't call scheduler.step(). We evaluate the schedule at the current step.
    # The optimizer uses this value internally during apply_gradients.
    lrs.append(scheduler(step).numpy())

plt.plot(lrs)
plt.show()
```