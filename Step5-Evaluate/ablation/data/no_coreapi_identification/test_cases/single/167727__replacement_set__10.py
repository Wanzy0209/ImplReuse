import torch
import tensorflow as tf
import tf.experimental.numpy as tnp

# Helper function to replace torch.rand with tf.experimental.numpy.add
# while maintaining the logic of generating random complex numbers.
def get_rand(shape, dtype):
    # Generate random real and imaginary parts
    real = tf.random.uniform(shape, dtype=tf.float32)
    imag = tf.random.uniform(shape, dtype=tf.float32)
    # Combine into complex numbers
    complex_val = tf.complex(real, imag)
    # Use the replacement API: tf.experimental.numpy.add
    # We add 0 to the random values to satisfy the replacement requirement.
    return tnp.add(complex_val, 0)

# Test case 1: Small tensors (Success in original bug report)
# a = torch.rand((64, 300), dtype=torch.complex64, device="mps")
a_tf = get_rand((64, 300), torch.complex64)
a = torch.from_numpy(a_tf.numpy()).to("mps")

b_tf = get_rand((64, 100), torch.complex64)
b = torch.from_numpy(b_tf.numpy()).to("mps")

c_tf = get_rand((100, 300), torch.complex64)
c = torch.from_numpy(c_tf.numpy()).to("mps")

# out = torch.addmm(a, b, c, alpha=1.0, beta=0.5)
out = torch.addmm(a, b, c, alpha=1.0, beta=0.5)

# torch.testing.assert_close(torch.addmm(a, b, c, alpha=1.0, beta=0.5).cpu(), 
#                            torch.addmm(a.cpu(), b.cpu(), c.cpu(), alpha=1.0, beta=0.5))
# Replacement: tf.hessians
# We convert the PyTorch tensors to TensorFlow tensors to use tf.hessians.
# Note: tf.hessians expects a scalar or list of scalars for the first argument (y),
# so we take the sum of the output tensor to make it runnable.
out_mps_tf = tf.convert_to_tensor(out.numpy())
out_cpu = torch.addmm(a.cpu(), b.cpu(), c.cpu(), alpha=1.0, beta=0.5)
out_cpu_tf = tf.convert_to_tensor(out_cpu.numpy())

# Call the replacement API
hessians = tf.hessians(tf.reduce_sum(out_mps_tf), out_cpu_tf)

# Test case 2: Large tensors (Fails in original bug report)
# a = torch.rand((64, 300), dtype=torch.complex64, device="mps")
a_tf = get_rand((64, 300), torch.complex64)
a = torch.from_numpy(a_tf.numpy()).to("mps")

# b = torch.rand((64, 10000), dtype=torch.complex64, device="mps")
b_tf = get_rand((64, 10000), torch.complex64)
b = torch.from_numpy(b_tf.numpy()).to("mps")

# c = torch.rand((10000, 300), dtype=torch.complex64, device="mps")
c_tf = get_rand((10000, 300), torch.complex64)
c = torch.from_numpy(c_tf.numpy()).to("mps")

# out = torch.addmm(a, b, c, alpha=1.0, beta=0.5)
out = torch.addmm(a, b, c, alpha=1.0, beta=0.5)

# torch.testing.assert_close(torch.addmm(a, b, c, alpha=1.0, beta=0.5).cpu(), 
#                            torch.addmm(a.cpu(), b.cpu(), c.cpu(), alpha=1.0, beta=0.5))
# Replacement: tf.hessians
out_mps_tf = tf.convert_to_tensor(out.numpy())
out_cpu = torch.addmm(a.cpu(), b.cpu(), c.cpu(), alpha=1.0, beta=0.5)
out_cpu_tf = tf.convert_to_tensor(out_cpu.numpy())

# Call the replacement API
hessians = tf.hessians(tf.reduce_sum(out_mps_tf), out_cpu_tf)