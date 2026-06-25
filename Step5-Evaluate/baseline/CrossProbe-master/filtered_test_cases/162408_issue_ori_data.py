import torch
torch.set_default_dtype(torch.bfloat16)
seqlens = torch.tensor([434, 78, 101, 411], dtype=torch.int32)
cu_seqlens = torch.cat((torch.zeros(1), torch.cumsum(seqlens, dim=0))).to(torch.int32)
print(cu_seqlens)  # Expected: [0, 434, 512, 613, 1024], Got: [0, 434, 512, 612, 1024]