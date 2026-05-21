import os

os.environ["TORCH_LOGS"] = "output_code"

import torch
import torch.nn

device = "cuda"

def slide_with_prod(state, new_events, arange, dev_null):
    batch_size, _, _ = new_events.shape

    concatenated = torch.cat([state, new_events], dim=1)

    # these three lines are a very complicated identity transformation:
    batch_idx = torch.arange(batch_size, dtype=torch.int32, device=device)[:, None]
    arange = arange[None, :]
    concatenated = concatenated[batch_idx, arange]

    state[:, :, :] = concatenated[:, -2048:, :]
    dev_null[:, :, :] = concatenated[:, :, :]

    # Use torch.prod to verify the state. 
    # Since the state tensor should contain zeros in the first 2048-2 columns,
    # the product of the entire tensor should be 0.
    # If the miscompilation occurs (non-zero values appear in the zeroed region),
    # torch.prod will return a non-zero value.
    return torch.prod(state)

# Compile the function using torch.compile
slide_with_prod_compiled = torch.compile(slide_with_prod)

# Initialize inputs
state = torch.zeros([4, 2048, 1024], device=device)
new_events = torch.arange(start=1, end=3, device=device)[None, :, None].expand(4, 2, 1024).contiguous()
arange = torch.arange(2050, dtype=torch.int32, device=device)
dev_null = torch.zeros([4, 2050, 1024], device=device)

# Run the eager version to get the expected result
expected_prod = slide_with_prod(
    state.clone(),
    new_events,
    arange,
    dev_null.clone()
)

print(f"Expected torch.prod result (eager): {expected_prod}")
assert expected_prod == 0.0, "Eager mode failed: state should contain zeros, making the product 0."

# Run the compiled version multiple times to catch potential race conditions/miscompilations
# The original bug report noted this could take a few attempts to exhibit.
for attempt in range(1, 100):
    state_input = torch.zeros([4, 2048, 1024], device=device)
    actual_prod = slide_with_prod_compiled(
        state_input,
        new_events,
        arange,
        dev_null.clone()
    )

    if actual_prod != expected_prod:
        print(f"Mismatch found on attempt {attempt}")
        print(f"Expected: {expected_prod}, Actual: {actual_prod}")
        # If the product is not 0, it means non-zero data leaked into the zeroed region
        raise AssertionError(
            f"torch.compile miscompilation detected via torch.prod: "
            f"expected product 0.0 but got {actual_prod}"
        )

print("Test passed: torch.compile behaves correctly with torch.prod.")