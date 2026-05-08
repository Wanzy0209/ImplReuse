File "/data/users/ngimel/pytorch/torch/distributed/_symmetric_memory/_nvshmem_triton.py", line 835, in broadcastmem_block_extern_wrapper
    return core.extern_elementwise(
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ngimel/.conda/envs/pytorch_3.12/lib/python3.12/site-packages/triton/language/core.py", line 43, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/ngimel/.conda/envs/pytorch_3.12/lib/python3.12/site-packages/triton/language/core.py", line 3374, in extern_elementwise
    ret_type = arg_type_symbol_dict[arg_types][1]
               ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^
KeyError: (triton.language.int32 ,triton.language.int64 ,triton.language.int64 ,triton.language.int32 ,triton.language.int32)
The above exception was the direct cause of the following exception:

triton.compiler.errors.CompilationError: at 30:11:
        - This is a collective operation - all PEs in the team must participate.
        - Must be called from kernels launched with cooperative launch.

    Example:
        ```
        # Broadcast 100 elements from PE 0 to all PEs
        nvshmem.broadcast(0, dest_tensor, src_tensor, 100, 0)
        ```
    """
    tl.static_assert(dest.type == source.type)
    nbytes = nelems * dest.type.element_ty.itemsize
    return broadcastmem_block_extern_wrapper(
           ^
(triton.language.int32 ,triton.language.int64 ,triton.language.int64 ,triton.language.int32 ,triton.language.int32)