import os

os.environ["TORCH_LOGS"] = "output_code"

import torch
import torch.nn

device = "cuda"

def slide_to_the_left1(state, new_events, arange, dev_null):
    _ = arange
    _ = dev_null

    concatenated = torch.cat([state, new_events], dim=1)
    state[:, :, :] = concatenated[:, -2048:, :]

def slide_to_the_left2(state, new_events, arange, dev_null):
    batch_size, _, _ = new_events.shape

    concatenated = torch.cat([state, new_events], dim=1)

    # these three lines are a very complicated identity transformation:
    batch_idx = torch.arange(batch_size, dtype=torch.int32, device=device)[:, None]
    arange = arange[None, :]
    concatenated = concatenated[batch_idx, arange]

    state[:, :, :] = concatenated[:, -2048:, :]
    dev_null[:, :, :] = concatenated[:, :, :]

slide_to_the_left3 = torch.compile(slide_to_the_left2)

state = torch.zeros([4, 2048, 1024], device=device)
slide_to_the_left1(
    state,
    torch.arange(start=1, end=3, device=device)[None, :, None].expand(4, 2, 1024).contiguous(),
    torch.arange(2050, dtype=torch.int32, device=device),
    torch.zeros([4, 2050, 1024], device=device),
)
display("slide_to_the_left1", state)
# because new_events has shape [4, 2, 1024], only 2 new events should have been added to the state, and the rest should still be zeros:
assert (state[:, :-2, :] == 0).all(), torch.nonzero(state[:, :-2, :])

state = torch.zeros([4, 2048, 1024], device=device)
slide_to_the_left2(
    state,
    torch.arange(start=1, end=3, device=device)[None, :, None].expand(4, 2, 1024).contiguous(),
    torch.arange(2050, dtype=torch.int32, device=device),
    torch.zeros([4, 2050, 1024], device=device),
)
display("slide_to_the_left2", state)
# ditto
assert (state[:, :-2, :] == 0).all(), torch.nonzero(state[:, :-2, :])

# this third one has a race condition that can take a few attempts to be exhibited on a T4
# anecdotally it seems to happen more often on the bigger chips
# single digit attempts to exhibit this have always sufficed on the NVIDIA chips I've tried
for attempt in range(1, 1000):
    state = torch.zeros([4, 2048, 1024], device=device)
    slide_to_the_left3(
        state,
        torch.arange(start=1, end=3, device=device)[None, :, None].expand(4, 2, 1024).contiguous(),
        torch.arange(2050, dtype=torch.int32, device=device),
        torch.zeros([4, 2050, 1024], device=device),
    )

    if not (state[:, :-2, :] == 0).all():
        display(f"slide_to_the_left3 attempt {attempt}", state, torch.nonzero(state[:, :-2, :]))
        break