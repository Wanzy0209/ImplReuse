import torch
import sys

# Handle missing dependency for older PyTorch versions
try:
    from torch.optim.swa_utils import get_ema_multi_avg_fn
except ImportError:
    print("Skipping test: 'get_ema_multi_avg_fn' is not available in this version of PyTorch.")
    sys.exit(0)

# Define the function using the similar API
# get_ema_multi_avg_fn returns a function that updates parameters in-place.
# We wrap it to return a value so it can be used with vjp.
def f(ema_params, curr_params):
    updater = get_ema_multi_avg_fn()
    updater(ema_params, curr_params, None)
    return ema_params[0]

# Create sparse tensors to test layout preservation
# This mimics the setup in the original bug report
indices = torch.tensor([[0, 1, 1], [2, 0, 2]])
values = torch.tensor([1.0, 2.0, 3.0])
ema = torch.sparse_coo_tensor(indices, values)
curr = torch.sparse_coo_tensor(indices, values + 1.0)

print("Testing direct call...")
try:
    # Note: get_ema_multi_avg_fn uses foreach_lerp_ for floats, which may not support sparse tensors.
    # This test checks if the API handles sparse inputs correctly or if it fails.
    result = f([ema], [curr])
    print(f"Direct call succeeded. Result layout: {result.layout}")
except Exception as e:
    print(f"Direct call failed: {e}")

print("\nTesting inside vjp (checking for GradTrackingTensor/layout issues)...")
try:
    # The original bug occurred here: GradTrackingTensor did not copy the layout,
    # causing operations expecting sparse tensors to fail.
    vjp_fn = torch.func.vjp(f, [ema], [curr])
    print("vjp call succeeded")
except Exception as e:
    print(f"vjp call failed: {e}")