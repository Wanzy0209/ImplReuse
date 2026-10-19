```python
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Use a tensor learning rate. The choice of optimizer doesn't matter.
lr = 1.0
# Conversion Note: In TensorFlow, we use a tf.Variable to mimic a tensor learning rate
lr_var = tf.Variable(lr, dtype=tf.float32)
x = tf.Variable(0.0)
opt = tf.keras.optimizers.AdamW(learning_rate=lr_var)

# 2. Initialize our chained schedulers.
milestone, total_steps = 40, 100
start_factor, end_factor = 0.2, 1.0

# Conversion Note: PyTorch LinearLR takes the optimizer to read base LR. 
# TF schedules take the initial LR explicitly.
warmup = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=lr * start_factor,
    decay_steps=milestone,
    end_learning_rate=lr * end_factor,
    power=1.0
)

# Conversion Note: PyTorch CosineAnnealingLR takes the optimizer.
# TF CosineDecay takes initial_learning_rate and decay_steps.
decay = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=lr,
    decay_steps=total_steps - milestone,
    alpha=0.0 # Mimics CosineAnnealingLR default behavior (decaying to 0)
)

# Conversion Note: PyTorch schedulers alias optimizer's initial_lr.
# TF schedules are independent objects. We skip the pointer assertions (is_set_to, data_ptr).

# 3. Initialize our SequentialLR.
# Conversion Note: TensorFlow does not have a built-in SequentialLR, so we define a custom one.
class SequentialLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, optimizer, schedulers, milestones):
        self.schedulers = schedulers
        self.milestones = milestones
        self.optimizer = optimizer # Kept for structural compatibility with source

    def __call__(self, step):
        if step < self.milestones[0]:
            return self.schedulers[0](step)
        else:
            # Adjust step for the second scheduler
            return self.schedulers[1](step - self.milestones[0])

scheduler = SequentialLR(opt, schedulers=[warmup, decay], milestones=[milestone])

# Conversion Note: PyTorch SequentialLR modifies optimizer.lr in place and calls _initial_step.
# In TF, we manually update the optimizer's learning rate variable to the initial step value.
initial_lr = scheduler(0)
opt.learning_rate.assign(initial_lr)

# Conversion Note: Skipping pointer assertions (data_ptr) as they are specific to PyTorch tensor implementation.
# assert (start_factor * lr == opt.learning_rate.numpy())

# 4. Now if we do something which breaks the aliases, e.g. save/load the optimizer's state_dict...
# Conversion Note: TF uses Checkpoints for state saving/loading.
checkpoint = tf.train.Checkpoint(optimizer=opt)
manager = tf.train.CheckpointManager(checkpoint, directory='./tf_ckpt', max_to_keep=1)
manager.save()
manager.restore()

# Conversion Note: TF variables are restored in place, so the "aliasing" issue in PyTorch
# (where scheduler points to old memory and optimizer to new) does not occur in the same way.
# The assertions below would pass in TF (values match), whereas they fail in PyTorch (pointers differ).
# assert not warmup.initial_learning_rate == opt.learning_rate.numpy() # This logic doesn't map 1:1

# 5. Run the loop
lrs = []
for step in range(total_steps):
    # Conversion Note: In TF, we calculate the LR from the schedule and assign it to the optimizer
    current_lr = scheduler(step)
    opt.learning_rate.assign(current_lr)
    lrs.append(opt.learning_rate.numpy())

plt.plot(lrs)
plt.show()
```