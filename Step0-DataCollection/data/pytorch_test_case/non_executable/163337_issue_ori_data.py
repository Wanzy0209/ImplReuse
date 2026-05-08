# In rocwmma_fattn/FlashAttn.py
import torch
...
flash_attn_wmma = torch.utils.cpp_extension.load(
    name="flash_attn_wmma",
    sources=[
        "rocwmma_fattn/host.cpp",
        "rocwmma_fattn/kernel_fp16.hip",
        "rocwmma_fattn/kernel_bf16.hip",
    ],
    extra_hip_cflags=hip_flags,
    ...
)