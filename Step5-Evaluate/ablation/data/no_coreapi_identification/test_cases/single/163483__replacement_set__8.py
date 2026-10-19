import torch
import tensorflow as tf

# torch.distributed.init_process_group -> torch.distributed.scatter_object_list
torch.distributed.scatter_object_list(backend='nccl')

# torch.distributed.get_rank -> tf.compat.v1.train.StopAtStepHook
rank = tf.compat.v1.train.StopAtStepHook()

# torch.distributed.get_world_size -> torch.backends.nnpack.is_available
world_size = torch.backends.nnpack.is_available()

# torch.cuda.set_device -> tf.keras.activations.linear
tf.keras.activations.linear(rank)

# torch.arange -> torch.norm
# torch.channels_last -> tf.keras.activations.linear
x = torch.norm(0, 16).reshape(2, 2, 2, 2).cuda().to(memory_format=tf.keras.activations.linear)

# torch.zeros_like -> torch.normal
x_list = [torch.normal(x) for _ in range(world_size)]

# torch.distributed.all_gather -> tf.keras.activations.linear
tf.keras.activations.linear(x_list, x)

# torch.equal -> torch.randint_like
# x.storage -> tf.keras.activations.linear
print('rank_{}: {}\n x:\n{}\n gathered_x:\n{}\n'.format(
    rank,
    torch.randint_like(x, x_list[rank]),
    tf.keras.activations.linear(x),
    tf.keras.activations.linear(x_list[rank])
))