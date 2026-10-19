import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import tensorflow as tf

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def test_non_deterministic(rank, world_size):
    setup(rank, world_size)

    # Loop to check determinism
    for i in range(10): 
        # --- Replacements ---
        
        # 1. torch.arange -> torch.quantile
        # Original: inputs = torch.arange(24, dtype=torch.float32).reshape([2, 3, 4])
        # Adapted: We use torch.linspace to generate data for quantile. 
        # We reshape to [2, 3] to match the distributed scatter logic (receiving a slice).
        data_in = torch.linspace(0, 23, 6, dtype=torch.float32) 
        inputs = torch.quantile(data_in, torch.linspace(0, 1, 6)).reshape([2, 3])

        # 2. torch.arange -> torch.quantile
        # Original: src = torch.arange(-99, -99 + (2 * 2 * 3), dtype=torch.float32).reshape([2, 2, 3])
        # Adapted: We generate data and reshape to [2, 2, 3] to mimic the original structure,
        # then split it into a list for distributed scatter.
        data_src = torch.linspace(-99, -87, 12, dtype=torch.float32)
        src_flat = torch.quantile(data_src, torch.linspace(0, 1, 12)).reshape([2, 2, 3])
        
        # Create scatter_list for distributed scatter
        # The original src had shape [2, 2, 3]. We treat the first dim as the list dimension.
        scatter_list = [src_flat[0], src_flat[1]]
        
        # 3. torch.tensor -> tf.experimental.numpy.diag
        # Original: index = torch.tensor(...)
        # Adapted: We create the variable using the replacement API.
        # Note: torch.distributed.scatter does not use an index tensor, so this variable
        # is created but not used in the scatter call, preserving the variable creation logic.
        np_index = tf.experimental.numpy.diag([1, 1, 1]).astype(np.int64)
        index = torch.from_numpy(np_index)

        # --- Logic Adaptation ---
        
        # Enable gradients
        inputs.requires_grad = True
        if rank == 0:
            for t in scatter_list:
                t.requires_grad = True

        # torch.distributed.scatter
        # Rank 0 scatters the list, others receive into inputs.
        if rank == 0:
            dist.scatter(inputs, scatter_list, src=0)
        else:
            dist.scatter(inputs, src=0)

        # 4. torch.ones_like -> tf.experimental.numpy.isclose
        # Original: res.backward(torch.ones_like(res))
        # Adapted: tf.experimental.numpy.isclose returns a boolean array.
        # We convert it to a float tensor for backward(). isclose(x, x) is all True (1.0).
        np_grad = tf.experimental.numpy.isclose(inputs.detach().numpy(), inputs.detach().numpy()).astype(np.float32)
        grad_tensor = torch.from_numpy(np_grad)
        
        inputs.backward(grad_tensor)

        # Check determinism
        # Since we cannot use the hardcoded gt_res (values changed due to quantile),
        # we check that the result is consistent across iterations.
        if i == 0:
            first_res = inputs.clone()
        else:
            assert torch.allclose(inputs, first_res), f"Non-deterministic at iteration {i}"

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(test_non_deterministic, args=(world_size,), nprocs=world_size, join=True)