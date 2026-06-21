import os

os.environ["TORCH_LOGS"] = "output_code"

import torch
import torch.nn

# Fix: Handle environments where torch.compile is not available (PyTorch < 2.0)
if not hasattr(torch, 'compile'):
    print("Warning: torch.compile not found (PyTorch < 2.0). Mocking torch.compile to use eager execution.")
    torch.compile = lambda func: func

device = "cuda"

def slide_and_check_any_eager(state, new_events, arange, dev_null):
    batch_size, _, _ = new_events.shape

    concatenated = torch.cat([state, new_events], dim=1)

    # these three lines are a very complicated identity transformation:
    batch_idx = torch.arange(batch_size, dtype=torch.int32, device=device)[:, None]
    arange = arange[None, :]
    concatenated = concatenated[batch_idx, arange]

    # Use torch.any to check if there are any non-zero values in the slice we are about to write.
    # Since new_events contains non-zeros, this should be True.
    has_non_zero = torch.any(concatenated[:, -2048:, :] > 0)

    state[:, :, :] = concatenated[:, -2048:, :]
    dev_null[:, :, :] = concatenated[:, :, :]
    
    return has_non_zero

# Compile the function to test torch.any under torch.compile
slide_and_check_any_compiled = torch.compile(slide_and_check_any_eager)

# Setup inputs
state = torch.zeros([4, 2048, 1024], device=device)
new_events = torch.arange(start=1, end=3, device=device)[None, :, None].expand(4, 2, 1024).contiguous()
arange = torch.arange(2050, dtype=torch.int32, device=device)
dev_null = torch.zeros([4, 2050, 1024], device=device)

# Run eager to get expected result for torch.any
expected_any_result = slide_and_check_any_eager(state.clone(), new_events, arange, dev_null)
print(f"Eager torch.any result: {expected_any_result}")

# Loop to catch potential miscompilation involving torch.any or the state update
for attempt in range(1, 1000):
    state_c = torch.zeros([4, 2048, 1024], device=device)
    dev_null_c = torch.zeros([4, 2050, 1024], device=device)
    
    compiled_any_result = slide_and_check_any_compiled(state_c, new_events, arange, dev_null_c)
    
    # Verify torch.any output correctness
    if compiled_any_result != expected_any_result:
        print(f"Bug detected in torch.any at attempt {attempt}: Expected {expected_any_result}, got {compiled_any_result}")
        break

    # Verify state tensor consistency (original bug context)
    # Only the last 2 rows should be non-zero because new_events has shape [4, 2, 1024]
    if not (state_c[:, :-2, :] == 0).all():
        print(f"Miscompilation detected in state tensor at attempt {attempt}")
        print("Non-zero indices in old state:", torch.nonzero(state_c[:, :-2, :]))
        break
else:
    print("Test passed (no miscompilation detected in 1000 attempts)")