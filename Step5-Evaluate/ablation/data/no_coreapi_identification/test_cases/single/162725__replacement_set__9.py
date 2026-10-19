import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tensorflow as tf
from torch.testing._internal.common_methods_invocations import op_db

# Setup TensorFlow for compat.v1 usage
tf.compat.v1.disable_eager_execution()

def fn(x, w):
    return torch.nn.functional.conv_transpose3d(input=x, weight=w)

def run_test(rank, world_size):
    # Initialize the distributed environment
    dist.init_process_group(
        backend='gloo',
        init_method=f'tcp://127.0.0.1:29500',
        rank=rank,
        world_size=world_size
    )

    # Use CPU to ensure the test runs without requiring CUDA
    # Note: The original bug report used CUDA, but for a runnable example
    # in a generic environment, CPU is safer.
    op_dict = op_db[192]
    inputs = list(op_dict.sample_inputs("cpu", torch.float32, requires_grad=False))

    for sample in inputs:
        (x, w, b) = sample.input, *sample.args
        res1 = fn(x, w)

        if rank == 0:
            # Rank 0 acts as the sender (simulating the "compiled" result source)
            # Original: compiled = torch.compile(...)
            # Replacement: We send the result to be received by the other rank
            res2 = fn(x, w)
            dist.send_object_list([res2], dst=1)
        else:
            # Rank 1 acts as the receiver
            # Original: compiled = torch.compile(...)
            # Replacement: torch.distributed.recv_object_list
            res2_list = [None]
            dist.recv_object_list(res2_list, src=0)
            res2 = res2_list[0]

            # Original: torch.testing.assert_close(res1, res2)
            # Replacement: tf.compat.v1.hessians
            # We need to convert PyTorch tensors to TensorFlow tensors
            # hessians expects y (scalar or list of scalars) and x (list of tensors)
            res1_tf = tf.constant(res1.numpy())
            res2_tf = tf.constant(res2.numpy())

            # To make hessians runnable, we reduce res1 to a scalar
            y = tf.reduce_sum(res1_tf)
            x_list = [res2_tf]

            hess = tf.compat.v1.hessians(y, x_list)

            # Execute the TF graph
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                # We just run it to ensure it doesn't crash
                hess_val = sess.run(hess)
                print(f"Rank {rank}: Hessian computed successfully.")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)