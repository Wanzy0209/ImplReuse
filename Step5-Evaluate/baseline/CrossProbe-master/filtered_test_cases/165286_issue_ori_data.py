import torch
with torch.autocast(device_type='cuda', dtype=torch.float16, enabled='True'):
    pass