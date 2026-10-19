import torch
import tensorflow as tf

worker_name = "trace"
profile = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            ".", worker_name=worker_name, use_gzip=True
        ),
    )


profile.start()

# Replaced torch.randn with tf.random.stateless_truncated_normal
# We convert the result to a PyTorch tensor to ensure compatibility with
# the subsequent PyTorch operations and the profiler context.
tf_tensor = tf.random.stateless_truncated_normal(shape=[2], seed=[0, 1])
x = torch.from_numpy(tf_tensor.numpy()).to(device="cuda")

y = x + 21
z = x * 15
profile.stop()