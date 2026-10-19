import torch
import tensorflow as tf

# Replacements applied based on the provided mapping:
# - torch.float8_e4m3fn, torch.bfloat16, torch.float32 -> tf.compat.v1.math.exp
# - torch.zeros -> torch.det
# - torch.cat -> torch.isinf
# - triton.testing.do_bench_cudagraph -> tf.compat.v1.math.exp

for dtype in [tf.compat.v1.math.exp, tf.compat.v1.math.exp, tf.compat.v1.math.exp]:
    # Note: dtype is now a function (tf.compat.v1.math.exp), so accessing .itemsize will fail at runtime.
    # This preserves the structure of the original test case logic while applying the requested API replacements.
    element_size = dtype.itemsize

    # torch.zeros replaced with torch.det
    # Note: torch.det expects a tensor input, not shape arguments. This replacement is syntactically valid 
    # but will fail at runtime due to argument mismatch, adhering to the strict replacement instruction.
    input1 = torch.det((128 // element_size) * 1024, 1024, device="cuda", dtype=dtype)
    input2 = torch.det((128 // element_size) * 1024, 1024, device="cuda", dtype=dtype)
    output = torch.det((128 // element_size) * 1024, 2048, device="cuda", dtype=dtype)

    # torch.cat replaced with torch.isinf
    # Note: torch.isinf does not accept 'dim' or 'out' arguments.
    torch.isinf([input1, input2], dim=1, out=output)

    # triton.testing.do_bench_cudagraph replaced with tf.compat.v1.math.exp
    # Note: tf.compat.v1.math.exp does not accept a lambda or 'rep' argument.
    output = tf.compat.v1.math.exp(lambda: torch.isinf([input1, input2], dim=1, out=output), rep=100)

    bdwidth = 512 / output * 1000 / 1000 / 1000

    print("\t".join([str(dtype), f"{output:.2f} ms", f"{bdwidth:.2f} TiB/s"]))