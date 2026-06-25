import torch
import gc
for _ in range(1000):
    t = torch.tensor([1,2,3])
    # Trigger pybind11 casting
    _ = str(t)
gc.collect()
# Check memory usage increase