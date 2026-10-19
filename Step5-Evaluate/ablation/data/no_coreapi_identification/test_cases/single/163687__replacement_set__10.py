import torch
import tensorflow as tf
from torch.nn.attention.flex_attention import flex_attention

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

# Helper to generate inputs using the requested replacement API: tf.experimental.numpy.add
def create_input(shape):
    # We use tf.experimental.numpy.add to generate the tensor data as requested.
    # Adding zeros and ones results in a tensor of ones.
    a = tf.experimental.numpy.zeros(shape, dtype=tf.float32)
    b = tf.experimental.numpy.ones(shape, dtype=tf.float32)
    # The replacement API call
    tf_tensor = tf.experimental.numpy.add(a, b)
    
    # Convert to PyTorch tensor and move to CUDA
    pt_tensor = torch.from_numpy(tf_tensor.numpy()).cuda()
    pt_tensor.requires_grad_(True)
    return pt_tensor

# Define shapes based on the original test case
input_shapes = [
    [27, 26, 62, 122],
    [27, 26, 124, 122],
    [27, 26, 124, 122],
    [27, 26, 124, 122],
    [27, 26, 248, 122],
    [27, 26, 248, 122],
    [27, 26, 31, 122],
    [27, 26, 124, 122],
    [27, 26, 31, 122],
    [27, 26, 124, 122],
    [27, 26, 124, 122]
]

# Generate arguments
args = [create_input(s) for s in input_shapes]

# Test Eager execution
print("Testing eager execution...")
try:
    out_eager = foo(*args)
    print("Eager execution successful.")
except Exception as e:
    print(f"Eager execution failed: {e}")

# Test Compiled execution
print("Testing compiled execution...")
try:
    compiled_foo = torch.compile(foo)
    out_compile = compiled_foo(*args)
    print("Compiled execution successful.")
    
    # Verify results match if both succeeded
    if 'out_eager' in locals():
        # Note: Due to potential precision differences or non-determinism in flex_attention,
        # we check closeness.
        assert torch.allclose(out_eager, out_compile, rtol=1e-2, atol=1e-2), "Outputs differ"
        print("Outputs match.")
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("OOM detected in compiled mode (Bug reproduced).")
    else:
        raise e