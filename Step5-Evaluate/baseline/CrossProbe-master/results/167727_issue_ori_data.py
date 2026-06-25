```python
import tensorflow as tf

# Helper function to mimic torch.rand for complex64
# torch.rand generates [0, 1) for both real and imaginary parts
def rand_complex(shape, device):
    with tf.device(device):
        real = tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32)
        imag = tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32)
        return tf.complex(real, imag)

# Helper function to mimic torch.addmm
# output = beta * input + alpha * (mat1 @ mat2)
def addmm(input_tensor, mat1, mat2, alpha=1.0, beta=1.0):
    return beta * input_tensor + alpha * tf.matmul(mat1, mat2)

# Use GPU as the equivalent to 'mps' (accelerator)
device_acc = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

# success
a = rand_complex((64, 300), device=device_acc)
b = rand_complex((64, 100), device=device_acc)
c = rand_complex((100, 300), device=device_acc)
out = addmm(a, b, c, alpha=1.0, beta=0.5)

# Mimic .cpu() by copying tensors to CPU context
with tf.device('/CPU:0'):
    a_cpu = tf.identity(a)
    b_cpu = tf.identity(b)
    c_cpu = tf.identity(c)
    # Move result to CPU to match source logic
    out_acc_cpu = tf.identity(out)

# Compare result on accelerator (moved to CPU) vs result on CPU
tf.debugging.assert_near(out_acc_cpu, 
                         addmm(a_cpu, b_cpu, c_cpu, alpha=1.0, beta=0.5))

# fails
a = rand_complex((64, 300), device=device_acc)
b = rand_complex((64, 10000), device=device_acc)
c = rand_complex((10000, 300), device=device_acc)
out = addmm(a, b, c, alpha=1.0, beta=0.5)

with tf.device('/CPU:0'):
    a_cpu = tf.identity(a)
    b_cpu = tf.identity(b)
    c_cpu = tf.identity(c)
    out_acc_cpu = tf.identity(out)

tf.debugging.assert_near(out_acc_cpu, 
                         addmm(a_cpu, b_cpu, c_cpu, alpha=1.0, beta=0.5))
```