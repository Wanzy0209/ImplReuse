import os

os.environ["TORCH_LOGS"] = "output_code"

import torch
import torch.nn

device = "cuda"

def f(x, y):
    y2 = torch.cat(
        [
            x[:, 1:],
            y[:, None] + 32 * 2048,
        ],
        dim=1,
    )

    x2 = x[:, 1:, None]
    y3 = y2[:, -1:, None]

    # Adapted to use torch.prod instead of the specific reshape/add logic
    # to verify the similar API in the context of the bug.
    combined = torch.cat([x2, y3], dim=1)
    return torch.prod(combined)

# This succeeds (eager mode)
print("Eager execution:")
print(f(
    torch.zeros(1, 32, dtype=torch.int64, device=device),
    torch.zeros(1, dtype=torch.int32, device=device),
))

# This tests the similar API under torch.compile
print("Compiled execution:")
if hasattr(torch, 'compile'):
    print(torch.compile(f)(
        torch.zeros(1, 32, dtype=torch.int64, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
    ))
else:
    print("Skipped: torch.compile is not available in this version of PyTorch.")