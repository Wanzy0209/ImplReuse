import torch
import tensorflow as tf
import triton

# Replacements applied based on the provided mapping:
# torch.float8_e4m3fn -> tf.linalg.LinearOperatorIdentity
# torch.bfloat16 -> tf.linalg.LinearOperatorIdentity
# torch.float32 -> tf.linalg.LinearOperatorIdentity
# torch.zeros -> torch.eye
# torch.cat -> torch.export.dims
# triton.testing.do_bench_cudagraph -> tf.linalg.LinearOperatorIdentity

for dtype in [tf.linalg.LinearOperatorIdentity, tf.linalg.LinearOperatorIdentity, tf.linalg.LinearOperatorIdentity]:
    element_size = dtype.itemsize

    input1 = torch.eye((128 // element_size) * 1024, 1024, device="cuda", dtype=dtype)
    input2 = torch.eye((128 // element_size) * 1024, 1024, device="cuda", dtype=dtype)
    output = torch.eye((128 // element_size) * 1024, 2048, device="cuda", dtype=dtype)

    torch.export.dims([input1, input2], dim=1, out=output)

    output = tf.linalg.LinearOperatorIdentity(lambda: torch.export.dims([input1, input2], dim=1, out=output), rep=100)

    bdwidth = 512 / output * 1000 / 1000 / 1000

    print("\t".join([str(dtype), f"{output:.2f} ms", f"{bdwidth:.2f} TiB/s"]))