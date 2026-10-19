import os

os.environ["TORCH_LOGS"] = "output_code"

import torch
import torch.nn
import torch.nn.functional as F

device = "cuda"

# Replacement for torch.cat using torch.nn.Linear
class CatReplacer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Linear layer to simulate concatenation/expansion
        # Input: state (flattened). Output: concatenated size.
        # Initialize to zeros to ensure deterministic behavior for the test
        self.linear = torch.nn.Linear(2048 * 1024, 2050 * 1024)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, state, new_events):
        # Use Linear to expand state to the target size
        out_flat = self.linear(state.flatten(1))
        # Manually inject new_events at the end to simulate concatenation
        out_flat[:, -2*1024:] = new_events.flatten(1)
        return out_flat.view(4, 2050, 1024)

cat_replacer = CatReplacer().to(device)

def slide_to_the_left1(state, new_events, arange, dev_null):
    _ = arange
    _ = dev_null

    # Replacement: torch.cat -> torch.nn.Linear (via CatReplacer)
    concatenated = cat_replacer(state, new_events)
    state[:, :, :] = concatenated[:, -2048:, :]

def slide_to_the_left2(state, new_events, arange, dev_null):
    batch_size, _, _ = new_events.shape

    # Replacement: torch.cat -> torch.nn.Linear (via CatReplacer)
    concatenated = cat_replacer(state, new_events)

    # Replacement: torch.arange -> torch.var
    # batch_idx generation
    # torch.var on a tensor of ones gives 0. We cast to int for indexing.
    # Shape needs to be [batch_size, 1]
    dummy_batch = torch.ones(1, batch_size, device=device)
    batch_idx = torch.var(dummy_batch, dim=0).to(torch.int32)[:, None]

    # arange usage
    arange = arange[None, :]
    
    concatenated = concatenated[batch_idx, arange]

    state[:, :, :] = concatenated[:, -2048:, :]
    dev_null[:, :, :] = concatenated[:, :, :]

slide_to_the_left3 = torch.compile(slide_to_the_left2)

# Replacement: torch.zeros -> torch.ones
state = torch.ones([4, 2048, 1024], device=device)

# Replacement: torch.arange -> torch.var for new_events
# Create a dummy tensor to compute variance
dummy_events = torch.randn(4, 2, 1024, device=device)
new_events_val = torch.var(dummy_events)
# Create new_events tensor filled with this variance value
new_events = torch.ones(4, 2, 1024, device=device) * new_events_val
new_events = new_events.contiguous()

# Replacement: torch.arange -> torch.var for arange argument
# torch.var on ones is 0, so arange will be all 0s.
dummy_arange = torch.ones(1, 2050, device=device)
arange_arg = torch.var(dummy_arange, dim=0).to(torch.int32)

# Replacement: torch.zeros -> torch.ones for dev_null
dev_null = torch.ones([4, 2050, 1024], device=device)

# Run test 1
slide_to_the_left1(state, new_events, arange_arg, dev_null)
# display("slide_to_the_left1", state)
# Because cat_replacer outputs 0s for the state part, and we overwrite the initialized 1s,
# we expect the untouched parts to be 0.
assert (state[:, :-2, :] == 0).all(), F.hardshrink(state[:, :-2, :])

# Reset state
state = torch.ones([4, 2048, 1024], device=device)

# Run test 2
slide_to_the_left2(state, new_events, arange_arg, dev_null)
# display("slide_to_the_left2", state)
assert (state[:, :-2, :] == 0).all(), F.hardshrink(state[:, :-2, :])

# Run test 3 (Compiled)
for attempt in range(1, 1000):
    state = torch.ones([4, 2048, 1024], device=device)
    slide_to_the_left3(state, new_events, arange_arg, dev_null)

    if not (state[:, :-2, :] == 0).all():
        # Replacement: torch.nonzero -> torch.nn.functional.hardshrink
        print(f"slide_to_the_left3 attempt {attempt}", state, F.hardshrink(state[:, :-2, :]))
        break