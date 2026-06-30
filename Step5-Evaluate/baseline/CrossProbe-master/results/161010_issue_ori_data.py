import tensorflow as tf
import tensorflow.experimental.numpy as tnp

# Check for GPU availability
# PyTorch: torch.cuda.is_available()
gpus = tf.config.list_physical_devices('GPU')
device_name = '/GPU:0' if gpus else '/CPU:0'

with tf.device(device_name):
    # PyTorch: torch.rand(5, 5, device=...)
    A = tf.random.uniform((5, 5))

def f(A, count):
    # PyTorch: Q, R = torch.linalg.qr(A)
    Q, R = tf.linalg.qr(A)

    # PyTorch: rhs = torch.ones(Q.shape[0], 1, device=A.device)
    rhs = tf.ones((tf.shape(Q)[0], 1), dtype=A.dtype)

    # PyTorch: a = torch.linalg.solve_triangular(R, Q.T @ rhs, upper=True)
    # TF: tf.linalg.triangular_solve(R, tf.linalg.matmul(Q, rhs, transpose_a=True), lower=False)
    # Note: lower=False corresponds to upper=True in PyTorch
    matmul_res = tf.linalg.matmul(Q, rhs, transpose_a=True)
    a = tf.linalg.triangular_solve(R, matmul_res, lower=False)

    # PyTorch: if a.stride() == a.clone(memory_format=torch.preserve_format).stride():
    # TF: Access strides via experimental numpy. tf.identity acts as clone.
    # Note: In TF, tensors are generally immutable, so 'clone' is identity.
    # We check if the strides of the result match the strides of an identity copy.
    stride_a = tnp.asarray(a).strides
    stride_clone = tnp.asarray(tf.identity(a)).strides

    if stride_a == stride_clone:
        return count + 1
    return count

# PyTorch: res1 = f(A, torch.zeros(1))
# Initialize count as float tensor to match PyTorch default behavior
count_init = tf.zeros(1, dtype=tf.float32)
res1 = f(A, count_init)
print(res1)

# PyTorch: res2 = torch.compile(f)(A, torch.zeros(1))
# TF: tf.function(f) is the equivalent of compiling a function
res2 = tf.function(f)(A, count_init)
print(res2)