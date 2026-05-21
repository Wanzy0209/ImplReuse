import os
import torch
from torch import library
from torch.testing._internal.common_utils import TestCase

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the custom operator name
qualname = "testlib::slide_to_the_left_custom"

# Define the operator schema
library.define(qualname + "(Tensor state, Tensor new_events, Tensor arange, Tensor dev_null) -> ()")

# Concrete implementation (mimicking slide_to_the_left2)
def slide_to_the_left_impl(state, new_events, arange, dev_null):
    batch_size, _, _ = new_events.shape

    concatenated = torch.cat([state, new_events], dim=1)

    # these three lines are a very complicated identity transformation:
    batch_idx = torch.arange(batch_size, dtype=torch.int32, device=state.device)[:, None]
    arange_expanded = arange[None, :]
    concatenated = concatenated[batch_idx, arange_expanded]

    state[:, :, :] = concatenated[:, -2048:, :]
    dev_null[:, :, :] = concatenated[:, :, :]

# Register the concrete implementation
library.impl(qualname, slide_to_the_left_impl)

# Register the abstract implementation using the similar API: torch.library.impl_abstract
# This defines the behavior for FakeTensors (used during compilation/tracing).
@library.impl_abstract(qualname)
def slide_to_the_left_abstract(state, new_events, arange, dev_null):
    # In the abstract implementation, we perform the same operations
    # to ensure shape and metadata inference is correct.
    # Since standard torch ops support FakeTensors, we can reuse the logic.
    batch_size, _, _ = new_events.shape
    
    # FakeTensor execution of cat
    concatenated = torch.cat([state, new_events], dim=1)
    
    # FakeTensor execution of indexing
    batch_idx = torch.arange(batch_size, dtype=torch.int32, device=state.device)[:, None]
    arange_expanded = arange[None, :]
    concatenated = concatenated[batch_idx, arange_expanded]
    
    # FakeTensor execution of in-place copy
    # Note: copy_ is the functional equivalent of in-place assignment for FakeTensors
    state.copy_(concatenated[:, -2048:, :])
    dev_null.copy_(concatenated[:, :, :])

# Wrapper function to be compiled
def run_custom_slide(state, new_events, arange, dev_null):
    # Call the custom operator
    torch.ops.testlib.slide_to_the_left_custom(state, new_events, arange, dev_null)

# Compile the wrapper
run_custom_slide_compiled = torch.compile(run_custom_slide)

# Test Case
def test_impl_abstract_compile():
    state = torch.zeros([4, 2048, 1024], device=device)
    new_events = torch.arange(start=1, end=3, device=device)[None, :, None].expand(4, 2, 1024).contiguous()
    arange = torch.arange(2050, dtype=torch.int32, device=device)
    dev_null = torch.zeros([4, 2050, 1024], device=device)

    # Run the compiled version
    # We run it multiple times to check for consistency/race conditions mentioned in the bug
    for attempt in range(10):
        state = torch.zeros([4, 2048, 1024], device=device)
        run_custom_slide_compiled(state, new_events, arange, dev_null)

        # Verify the state tensor is correct
        # Only the last 2 rows should be non-zero
        assert (state[:, :-2, :] == 0).all(), f"Bug reproduced at attempt {attempt}: nonzero found in state[:, :-2, :]"
        
        # Verify the last 2 rows have the expected values
        # new_events contains values 1 and 2 expanded to 1024
        assert (state[:, -2, :] == 1).all(), "Value mismatch at row -2"
        assert (state[:, -1, :] == 2).all(), "Value mismatch at row -1"

    print("Test passed: torch.library.impl_abstract registration works correctly with torch.compile")

if __name__ == "__main__":
    test_impl_abstract_compile()