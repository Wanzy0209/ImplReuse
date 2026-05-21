import os

os.environ["TORCH_LOGS"] = "output_code"

import torch
import torch.nn

device = "cuda"

# Adapted function using torch.all (the similar API)
# This function mimics the structure of the original reproducer but uses torch.all
# to verify if the compilation issue affects this API as well.
def g(x, y):
    y2 = torch.cat(
        [
            x[:, 1:],
            y[:, None] + 32 * 2048,
        ],
        dim=1,
    )

    x2 = x[:, 1:, None]
    y3 = y2[:, -1:, None]

    # Calculate the intermediate tensor
    res = (
        torch.cat([x2, y3], dim=1)
        + torch.arange(-2048, 0, device=device)[None, None, :]
    ).reshape(1, 32 * 2048)
    
    # Use torch.all on the result
    return torch.all(res > -10000)

# This succeeds (eager execution)
print("Eager execution:")
print(g(
    torch.zeros(1, 32, dtype=torch.int64, device=device),
    torch.zeros(1, dtype=torch.int32, device=device),
))

# This tests compilation with torch.all
print("Compiled execution:")
print(torch.compile(g)(
    torch.zeros(1, 32, dtype=torch.int64, device=device),
    torch.zeros(1, dtype=torch.int32, device=device),
))