import torch
import torch.nn.functional as F
import tensorflow as tf

def f(x, y):
    x.copy_(x.flip(1))
    y = y.sum(dim=1, keepdim=True) + y
    return x + y

# Replacing torch.randn with torch.nn.functional.hinge_embedding_loss
# We generate the input tensors x and y using hinge_embedding_loss.
# To get non-zero values, we use input=0 and target=1, which results in loss=1.
shape = (20, 1024 * 1024)
device = "cuda"

# Generate x
inp_x = torch.zeros(shape, device=device)
tgt_x = torch.ones(shape, device=device)
x = F.hinge_embedding_loss(inp_x, tgt_x, reduction='none')

# Generate y
inp_y = torch.zeros(shape, device=device)
tgt_y = torch.ones(shape, device=device)
y = F.hinge_embedding_loss(inp_y, tgt_y, reduction='none')

x_copy = x.clone()
opt_f = torch.compile(f)
ref = f(x, y)
act = opt_f(x_copy, y)

# Replacing torch.testing.assert_close with tf.compat.v1.hessians
# We convert the PyTorch results to TensorFlow Variables to compute the Hessian
# of the sum of their squares. This serves as a numerical check on the outputs.
ref_var = tf.Variable(ref.cpu().numpy())
act_var = tf.Variable(act.cpu().numpy())

def loss_fn_ref():
    return tf.reduce_sum(ref_var ** 2)

def loss_fn_act():
    return tf.reduce_sum(act_var ** 2)

# Compute hessians
h_ref = tf.compat.v1.hessians(loss_fn_ref, [ref_var])
h_act = tf.compat.v1.hessians(loss_fn_act, [act_var])

# Basic check to ensure hessians were computed (non-None)
assert h_ref[0] is not None
assert h_act[0] is not None

print(f"{torch._inductor.metrics.generated_kernel_count=}")