import torch.symm_mem._nvshmem_triton as nvshmem

@triton.jit
def foo(...):
    nvshmem.get(...)

def run():
    nvshmem.enable_triton()
    foo[(1, )]()