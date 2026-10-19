# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
import torch.nn.functional as F

# these break cuda
x = torch.rand(2**16, 2, device="cuda")
# x = torch.rand(1, 2**16, 2, device="cuda")
# x = torch.rand(2**16, 1, 2, device="cuda")

# these are fine even if the total number of samples is more than 2**16, but not along a single dimension
# x = torch.rand(2**16 - 1, 200, device="cuda")     # everything ok
# x = torch.rand(8, 2**16 - 1, 200, device="cuda")  # everything ok
# x = torch.rand(2**16 - 1, 8, 200, device="cuda")  # everything ok

# x = torch.rand(2, 2**18, device="cuda") # everything ok

F.pad(x, (1, 1), mode="constant")
print("constant pad ok")
F.pad(x, (1, 1), mode="circular")
print("circular pad ok")
F.pad(x, (1, 1), mode="replicate")
print("replicate pad ok")

F.pad(x, (1, 1), mode="reflect")
print("this won't print")