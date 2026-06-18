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

    return (
        torch.cat([x2, y3], dim=1)
        + torch.arange(-2048, 0, device=device)[None, None, :]
    ).reshape(1, 32 * 2048)

# This succeeds
display(
    f(
        torch.zeros(1, 32, dtype=torch.int64, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
    )
)

# This crashes
display(
    torch.compile(f)(
        torch.zeros(1, 32, dtype=torch.int64, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
    )
)