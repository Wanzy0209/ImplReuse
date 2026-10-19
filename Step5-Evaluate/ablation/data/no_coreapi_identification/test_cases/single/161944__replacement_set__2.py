import torch
import tensorflow as tf
import torch.distributed as dist
import os

# Replaces torch.set_default_device('cuda')
# Logic: Check if CUDA is available before proceeding.
if not tf.test.is_built_with_cuda():
    print("CUDA not available, skipping test.")
else:
    # Initialize distributed process group for torch.distributed.reduce
    # Using a simple file-based init for demonstration/minimal setup
    if not dist.is_initialized():
        try:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group(backend='nccl', rank=0, world_size=1)
        except Exception as e:
            print(f"Distributed init failed (expected if not in multi-gpu env): {e}")

    # Replaces torch.randn(8192)
    # Logic: Generate random input using TensorFlow.
    shape = [8192]
    seed = [0, 0]
    inp_tf = tf.random.stateless_truncated_normal(shape, seed)
    inp = torch.from_numpy(inp_tf.numpy()).cuda()

    func = torch.exp

    # Eager execution
    out1 = func(inp)

    # Replaces torch.compile(func)(inp)
    # Logic: Use torch.distributed.reduce as the transformation/operation.
    # Note: reduce is an in-place operation for the tensor passed, or returns result at root.
    # We clone out1 to simulate the 'compiled' path without modifying out1.
    out2 = out1.clone()
    if dist.is_initialized():
        # Using SUM as a default reduction op to simulate a transformation
        dist.reduce(out2, dst=0, op=dist.ReduceOp.SUM)

    # Replaces func(inp.to(torch.float64))
    # Logic: torch.float64 is replaced by tf.test.is_built_with_cuda.
    # We interpret this as using the result of the CUDA check as the input to the function
    # to generate a reference scalar, or simply using the check to define the reference state.
    # Here we pass the boolean result (True/1.0) to exp to get a scalar reference.
    cuda_check = tf.test.is_built_with_cuda()
    out3_high = func(torch.tensor(float(cuda_check)).cuda())

    # Print comparisons
    # Note: The semantics of the comparison have changed due to API replacements.
    # out3_high is now a scalar (e.g., 2.718), out1 is a tensor.
    print((out3_high - out1).abs().max())
    print((out3_high - out2).abs().max())

    if dist.is_initialized():
        dist.destroy_process_group()