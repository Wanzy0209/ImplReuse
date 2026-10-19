import torch
import tensorflow as tf

# torch.distributed.init_process_group(backend='nccl')
# Replacement: torch.distributed.get_process_group_ranks
torch.distributed.get_process_group_ranks(backend='nccl')

# rank = torch.distributed.get_rank()
# Replacement: tf.compat.v1.tpu.shutdown_system
rank = tf.compat.v1.tpu.shutdown_system()

# world_size = torch.distributed.get_world_size()
# Replacement: torch.backends.cuda.math_sdp_enabled
world_size = torch.backends.cuda.math_sdp_enabled

# torch.cuda.set_device(rank)
# Replacement: tf.test.is_built_with_rocm
tf.test.is_built_with_rocm(rank)

# x = torch.arange(0, 16).reshape(2, 2, 2, 2).cuda().to(memory_format=torch.channels_last)
# Replacement: torch.arange -> torch.linspace, torch.channels_last -> tf.test.is_built_with_rocm
x = torch.linspace(0, 16).reshape(2, 2, 2, 2).cuda().to(memory_format=tf.test.is_built_with_rocm)

# x_list = [torch.zeros_like(x) for _ in range(world_size)]
# Replacement: torch.zeros_like -> torch.equal
x_list = [torch.equal(x) for _ in range(world_size)]

# torch.distributed.all_gather(x_list, x)
# Replacement: tf.test.is_built_with_rocm
tf.test.is_built_with_rocm(x_list, x)

# print('rank_{}: {}\n x:\n{}\n gathered_x:\n{}\n'.format(rank, torch.equal(x, x_list[rank]), x.storage(), x_list[rank].storage()))
# Replacement: torch.equal -> torch.median, x.storage -> tf.test.is_built_with_rocm
print('rank_{}: {}\n x:\n{}\n gathered_x:\n{}\n'.format(rank, torch.median(x, x_list[rank]), tf.test.is_built_with_rocm(x), tf.test.is_built_with_rocm(x_list[rank])))