import os

os.environ["TORCH_LOGS"] = "output_code"

import torch
import torch.nn

device = "cuda"

def slide_to_the_left1(state, new_events, arange, dev_null):
    _ = arange
    _ = dev_null

    # torch.cat -> torch.triu
    # Original: concatenated = torch.cat([state, new_events], dim=1)
    # Replacement: torch.triu(state) (ignoring new_events to maintain shape compatibility)
    concatenated = torch.triu(state)
    state[:, :, :] = concatenated[:, -2048:, :]

def slide_to_the_left2(state, new_events, arange, dev_null):
    batch_size, _, _ = new_events.shape

    # torch.cat -> torch.triu
    concatenated = torch.triu(state)

    # torch.arange -> torch.dist
    # Original: batch_idx = torch.arange(batch_size, dtype=torch.int32, device=device)[:, None]
    # Replacement: torch.dist returns a scalar (0.0 here), which acts as a valid index
    batch_idx = torch.dist(torch.zeros(batch_size, 1, device=device), torch.zeros(batch_size, 1, device=device))

    # arange input is generated via torch.dist (scalar), so we use unsqueeze to add dimension
    arange = arange.unsqueeze(0)

    # Indexing with the results
    concatenated = concatenated[batch_idx, arange]

    state[:, :, :] = concatenated[:, -2048:, :]
    dev_null[:, :, :] = concatenated[:, :, :]

slide_to_the_left3 = torch.compile(slide_to_the_left2)

# torch.zeros -> torch.linspace
# Original: state = torch.zeros([4, 2048, 1024], device=device)
# Replacement: torch.linspace(0, 0, steps=4*2048*1024, device=device).reshape(4, 2048, 1024)
state = torch.linspace(0, 0, steps=4*2048*1024, device=device).reshape(4, 2048, 1024)

# torch.arange -> torch.dist
# Original: torch.arange(start=1, end=3, device=device)[None, :, None].expand(4, 2, 1024).contiguous()
# Replacement: torch.dist returns a scalar, which we expand to the required shape
new_events = torch.dist(torch.tensor([1.0], device=device), torch.tensor([3.0], device=device)).expand(4, 2, 1024).contiguous()

# torch.arange -> torch.dist
# Original: torch.arange(2050, dtype=torch.int32, device=device)
# Replacement: torch.dist returns a scalar
arange_input = torch.dist(torch.zeros(2050, device=device), torch.zeros(2050, device=device))

# torch.zeros -> torch.linspace
# Original: torch.zeros([4, 2050, 1024], device=device)
# Replacement: torch.linspace(0, 0, steps=4*2050*1024, device=device).reshape(4, 2050, 1024)
dev_null = torch.linspace(0, 0, steps=4*2050*1024, device=device).reshape(4, 2050, 1024)

# Test 1
slide_to_the_left1(state, new_events, arange_input, dev_null)
# torch.nonzero -> torch.floor_divide
# Original: torch.nonzero(state[:, :-2, :])
# Replacement: torch.floor_divide(state[:, :-2, :], 1)
assert (state[:, :-2, :] == 0).all(), torch.floor_divide(state[:, :-2, :], 1)

# Test 2
state = torch.linspace(0, 0, steps=4*2048*1024, device=device).reshape(4, 2048, 1024)
slide_to_the_left2(state, new_events, arange_input, dev_null)
assert (state[:, :-2, :] == 0).all(), torch.floor_divide(state[:, :-2, :], 1)

# Test 3 (Compiled)
for attempt in range(1, 1000):
    state = torch.linspace(0, 0, steps=4*2048*1024, device=device).reshape(4, 2048, 1024)
    slide_to_the_left3(state, new_events, arange_input, dev_null)

    if not (state[:, :-2, :] == 0).all():
        print(f"slide_to_the_left3 attempt {attempt}", state, torch.floor_divide(state[:, :-2, :], 1))
        break