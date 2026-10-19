```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 1. Use a learning rate schedule. TF optimizers take a schedule or float.
# Note: TF does not support passing a mutable tensor for LR like PyTorch.
# We define a custom schedule to mimic the behavior of the chained schedulers.
lr = 1.0
x = tf.Variable(0.0)

# Define the combined schedule logic here to pass to the optimizer.
# This mimics the SequentialLR + LinearLR + CosineAnnealingLR setup.
class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, total_steps, start_factor, end_factor):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.decay_steps = total_steps - warmup_steps

    def __call__(self, step):
        # Linear Warmup phase
        # PyTorch LinearLR: start_factor + (end_factor - start_factor) * (step / total_iters)
        # Multiplied by base_lr (initial_lr)
        warmup_lr = self.initial_lr * (self.start_factor + (self.end_factor - self.start_factor) * (step / self.warmup_steps))

        # Cosine Decay phase
        # PyTorch CosineAnnealingLR: eta_min + (base_lr - eta_min) * (1 + cos(pi * t / T_max)) / 2
        # t = step - warmup_steps, T_max = decay_steps, eta_min = 0
        decay_step = step - self.warmup_steps
        cosine_lr = self.initial_lr * 0.5 * (1 + tf.cos(tf.constant(np.pi) * decay_step / self.decay_steps))

        # Switch between warmup and decay
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

# 2. Initialize our chained schedulers parameters.
milestone, total_steps = 40, 100
start_factor, end_factor = 0.2, 1.0

# Create the schedule object
scheduler = WarmupCosineSchedule(lr, milestone, total_steps, start_factor, end_factor)

# Initialize Optimizer
opt = tf.keras.optimizers.AdamW(learning_rate=scheduler)

# Note that the schedule is attached to the optimizer.
# In PyTorch, schedulers alias with the optimizer's initial_lr tensor.
# In TF, the optimizer holds a reference to the schedule object.
assert opt.learning_rate is scheduler

# 3. Initialize our SequentialLR equivalent.
# The logic is encapsulated in the WarmupCosineSchedule class.

# SequentialLR.__init__ aliases our optimizer's lr and initial_lr.
# In TF, the learning rate is determined by calling the schedule.
# We verify the initial LR value matches the expected start_factor * lr.
initial_lr_val = opt.learning_rate(0).numpy()
assert abs(initial_lr_val - (start_factor * lr)) < 1e-6, f"Expected {start_factor * lr}, got {initial_lr_val}"

# It also calls _initial_step() on our LinearLR, setting all tensors to start_factor * lr.
# We verify this behavior by checking the schedule output at step 0.
assert abs(opt.learning_rate(0).numpy() - (start_factor * lr)) < 1e-6

# 4. Now if we do something which breaks the aliases, e.g. save/load the optimizer's state_dict...
# In TF, saving/loading weights does not break the object reference to the schedule.
# The behavior described in the PyTorch code (incorrect scaling after load) does not occur in TF
# because the schedule is a functional object, not a stateful tensor in the optimizer.

# Simulate save/load
weights = opt.get_weights()
opt.set_weights(weights)

# Our base_lrs are incorrect... (PyTorch comment)
# In TF, the schedule remains correct and attached.
# The assertion below checks for memory address changes which is not applicable in TF.
# assert not warmup.base_lrs[0].data_ptr() == opt.param_groups[0]['initial_lr'].data_ptr()

# The LinearLR will work as intended... but the outputs of further schedulers are incorrectly scaled...
# In TF, the schedule handles the transition correctly without manual step calls or state aliasing issues.
lrs = []
for step in range(total_steps):
    # In PyTorch: scheduler.step()
    # In TF: The optimizer uses the schedule automatically. 
    # To record the LR for the plot, we evaluate the schedule at the current step.
    current_lr = opt.learning_rate(step).numpy()
    lrs.append(current_lr)
    
    # Optional: Perform a dummy optimization step to update optimizer state
    # opt.apply_gradients([(tf.zeros_like(x), x)])

plt.plot(lrs)
plt.show()
```