import torch
import tensorflow as tf
import sys

# Enable eager execution to mimic PyTorch's default behavior
tf.compat.v1.enable_eager_execution()

MAX = 3
BATCH = 37

# Define inputs
# PyTorch: torch.randint(MAX, (BATCH,), dtype=torch.int64)
idxs = tf.random.uniform((BATCH,), minval=0, maxval=MAX, dtype=tf.int64)
# PyTorch: torch.rand((BATCH, MAX), dtype=torch.float64)
x = tf.random.uniform((BATCH, MAX), dtype=tf.float64)

def func(x, idxs):
    # PyTorch: x.square() * torch.nn.functional.one_hot(idxs, MAX)
    # TensorFlow: tf.square(x) * tf.one_hot(idxs, MAX)
    return tf.square(x) * tf.one_hot(idxs, MAX, dtype=x.dtype)

def jacfunc(x, idxs):
    # PyTorch: torch.func.jacfwd(func, argnums=(0,))(x, idxs)
    # TensorFlow: Use tf.GradientTape to compute the Jacobian
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = func(x, idxs)
    # tape.jacobian computes the full jacobian, similar to jacfwd for vector-valued functions
    return tape.jacobian(y, x)

# 1. Works (Eager execution)
print("Running eager execution...")
try:
    out_eager = jacfunc(x, idxs)
    print("Eager execution successful. Output shape:", out_eager.shape)
except Exception as e:
    print(f"Eager execution failed: {e}")
    sys.exit(1)

# 2. Test with tf.compat.v1.tpu.rewrite
# Note: tf.compat.v1.tpu.rewrite is the cross-library similar API to torch.compile.
# It compiles the computation for TPU (XLA).
# This requires a TPU environment to run successfully.
print("Running tf.compat.v1.tpu.rewrite...")

try:
    # Initialize TPU system (Required for rewrite)
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    
    # PyTorch: jacfunc = torch.compile(jacfunc, dynamic=True); out = jacfunc(x, idxs)
    # TensorFlow: out = tf.compat.v1.tpu.rewrite(jacfunc, inputs=[x, idxs])
    # rewrite takes the function and the inputs, compiles, and executes.
    out_compiled = tf.compat.v1.tpu.rewrite(jacfunc, inputs=[x, idxs])
    
    print("TPU rewrite successful. Output shape:", out_compiled.shape)
    
    # Assertion to check if compiled output matches eager output shape
    assert out_eager.shape == out_compiled.shape, "Shape mismatch between eager and compiled"
    
except tf.errors.NotFoundError:
    print("TPU device not found. Skipping TPU rewrite test (this is expected on non-TPU hardware).")
except Exception as e:
    print(f"TPU rewrite failed with error: {e}")
    # In the context of the original bug report, a failure here indicates a similar issue.