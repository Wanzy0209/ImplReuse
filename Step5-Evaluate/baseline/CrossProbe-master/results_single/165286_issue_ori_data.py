# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
from torch.cuda.amp import autocast
with torch.autocast(device_type='cuda', dtype=torch.float16, enabled = True):
                output_seq = model(input_seq)