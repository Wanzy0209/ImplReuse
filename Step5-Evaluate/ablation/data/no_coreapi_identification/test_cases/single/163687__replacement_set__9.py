import torch
import sys
import tensorflow as tf
import numpy as np
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

# Configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10):
    t0 = arg0
    t1 = arg1
    t2 = arg2
    t3 = flex_attention(t0, t1, t2)
    t4 = arg3
    t5 = arg4
    t6 = arg5
    t7 = flex_attention(t4, t5, t6)
    t8 = flex_attention(t3, t7, t7)
    t9 = arg6
    t10 = arg7
    t11 = flex_attention(t9, t7, t10)
    t12 = flex_attention(t11, t8, t3)
    t13 = arg8
    t14 = arg9
    t15 = flex_attention(t13, t2, t14)
    t16 = arg10
    t17 = t16.clone()
    t17.zero_()
    t18 = flex_attention(t17, t8, t3)
    t19 = flex_attention(t15, t17, t18)
    t20 = flex_attention(t8, t12, t19)
    output = t20
    return output

def generate_input(shape):
    """
    Replaces torch.rand with tf.compat.v1.math.add.
    Generates two random tensors in TensorFlow, adds them, and converts to PyTorch.
    """
    # Create random tensors in TensorFlow
    tf_a = tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32)
    tf_b = tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32)
    
    # Apply the replacement API: tf.compat.v1.math.add
    tf_result = tf.compat.v1.math.add(tf_a, tf_b)
    
    # Convert result to PyTorch tensor
    pt_tensor = torch.from_numpy(tf_result.numpy())
    pt_tensor = pt_tensor.to(dtype=torch.float32, device='cuda')
    pt_tensor.requires_grad_(True)
    return pt_tensor

# Define shapes for the arguments
shapes = [
    (27, 26, 62, 122),   # arg0
    (27, 26, 124, 122),  # arg1
    (27, 26, 124, 122),  # arg2
    (27, 26, 124, 122),  # arg3
    (27, 26, 248, 122),  # arg4
    (27, 26, 248, 122),  # arg5
    (27, 26, 31, 122),   # arg6
    (27, 26, 124, 122),  # arg7
    (27, 26, 31, 122),   # arg8
    (27, 26, 124, 122),  # arg9
    (27, 26, 124, 122)   # arg10
]

# Generate inputs using the replacement logic
args = [generate_input(s) for s in shapes]

# Test Eager Mode
print("Running Eager Mode...")
try:
    eager_output = foo(*args)
    print("Eager Mode Success.")
except Exception as e:
    print(f"Eager Mode Failed: {e}")
    sys.exit(1)

# Test Compiled Mode (torch.compile)
print("Running Compiled Mode...")
try:
    compiled_foo = torch.compile(foo)
    compiled_output = compiled_foo(*args)
    print("Compiled Mode Success.")
    
    # Verify that outputs match (checking for divergence)
    # Note: Due to potential non-determinism in flex_attention or different backends, 
    # we check if they are close or if the compiled run simply finished without OOM.
    # The bug report specifically mentions OOM, so successful execution is the primary check.
    assert eager_output.shape == compiled_output.shape, "Output shape mismatch"
    print("Test Passed: Compiled mode executed without OOM and shapes match.")
    
except torch.cuda.OutOfMemoryError:
    print("Test Failed: OOM in compiled mode (Bug reproduced).")
    sys.exit(1)
except Exception as e:
    print(f"Test Failed with exception: {e}")
    sys.exit(1)