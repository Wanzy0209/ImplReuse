import torch
x = torch.empty((0,), dtype=torch.float32)
if torch.backends.mps.is_available():
    mps_x = x.to('mps')
    print(torch.nanmedian(mps_x))