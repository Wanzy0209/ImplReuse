import rmm
    from rmm.allocators.torch import rmm_torch_allocator
    import torch

    rmm.reinitialize(pool_allocator=True)
    torch.cuda.memory.change_current_allocator(rmm_torch_allocator)