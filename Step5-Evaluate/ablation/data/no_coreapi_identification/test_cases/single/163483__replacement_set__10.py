import torch
import tensorflow as tf

# torch.distributed.init_process_group -> torch.nn.functional.binary_cross_entropy
torch.nn.functional.binary_cross_entropy(backend='nccl')

# torch.distributed.get_rank -> tf.experimental.dtensor.shutdown_accelerator_system
rank = tf.experimental.dtensor.shutdown_accelerator_system()

# torch.distributed.get_world_size -> torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed
world_size = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed

# torch.cuda.set_device -> tf.VariableSynchronization
tf.VariableSynchronization(rank)

# torch.arange -> torch.ones
# torch.channels_last -> tf.VariableSynchronization
x = torch.ones(16).reshape(2, 2, 2, 2).cuda().to(memory_format=tf.VariableSynchronization)

# torch.zeros_like -> torch.norm
x_list = [torch.norm(x) for _ in range(world_size)]

# torch.distributed.all_gather -> tf.VariableSynchronization
tf.VariableSynchronization(x_list, x)

# torch.equal -> torch.nn.functional.pixel_shuffle
# x.storage -> tf.VariableSynchronization
print('rank_{}: {}\n x:\n{}\n gathered_x:\n{}\n'.format(
    rank,
    torch.nn.functional.pixel_shuffle(x, x_list[rank]),
    tf.VariableSynchronization(x),
    tf.VariableSynchronization(x_list[rank])
))