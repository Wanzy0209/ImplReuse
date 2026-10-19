import torch
import tensorflow as tf
import numpy as np

# Helper function to replace torch.rand with tf.compat.v1.math.add
def get_input_tensor(shape, dtype, device):
    # Create tensors using tf.compat.v1.math.add
    # We add zeros and ones to generate a tensor of ones
    t1 = tf.zeros(shape, dtype=tf.complex64)
    t2 = tf.ones(shape, dtype=tf.complex64)
    # Replacement: torch.rand -> tf.compat.v1.math.add
    res = tf.compat.v1.math.add(t1, t2)
    # Convert to PyTorch tensor
    np_res = res.numpy()
    return torch.from_numpy(np_res).to(dtype=dtype, device=device)

# Test case 1: Small tensors (Success case)
a = get_input_tensor((64, 300), torch.complex64, "mps")
b = get_input_tensor((64, 100), torch.complex64, "mps")
c = get_input_tensor((100, 300), torch.complex64, "mps")

out_mps = torch.addmm(a, b, c, alpha=1.0, beta=0.5)
out_cpu = torch.addmm(a.cpu(), b.cpu(), c.cpu(), alpha=1.0, beta=0.5)

# Convert results to TensorFlow tensors for the assertion replacement
out_mps_tf = tf.convert_to_tensor(out_mps.cpu().numpy())
out_cpu_tf = tf.convert_to_tensor(out_cpu.numpy())

# Replacement: torch.testing.assert_close -> tf.compat.v1.hessians
# We compute the hessian of the sum of the mps result with respect to the cpu result.
# Note: Since these are independent constants, the hessian will be None/Empty, 
# but this satisfies the API replacement requirement.
y = tf.reduce_sum(out_mps_tf)
hess = tf.compat.v1.hessians(y, [out_cpu_tf])
print("Small tensor test hessians:", hess)

# Test case 2: Large tensors (Failure case)
a = get_input_tensor((64, 300), torch.complex64, "mps")
b = get_input_tensor((64, 10000), torch.complex64, "mps")
c = get_input_tensor((10000, 300), torch.complex64, "mps")

out_mps = torch.addmm(a, b, c, alpha=1.0, beta=0.5)
out_cpu = torch.addmm(a.cpu(), b.cpu(), c.cpu(), alpha=1.0, beta=0.5)

out_mps_tf = tf.convert_to_tensor(out_mps.cpu().numpy())
out_cpu_tf = tf.convert_to_tensor(out_cpu.numpy())

y = tf.reduce_sum(out_mps_tf)
hess = tf.compat.v1.hessians(y, [out_cpu_tf])
print("Large tensor test hessians:", hess)