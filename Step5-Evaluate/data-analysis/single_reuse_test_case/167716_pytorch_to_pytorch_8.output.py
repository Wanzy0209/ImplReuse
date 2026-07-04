import torch
from torch.optim import swa_utils

# Handle missing get_ema_multi_avg_fn for compatibility with older PyTorch versions
if not hasattr(swa_utils, 'get_ema_multi_avg_fn'):
    def get_ema_multi_avg_fn(decay=0.9):
        def ema_avg_fn(avg_p_list, p_t_list, _):
            for avg_p, p_t in zip(avg_p_list, p_t_list):
                avg_p.mul_(decay).add_(p_t, alpha=1 - decay)
        return ema_avg_fn
    swa_utils.get_ema_multi_avg_fn = get_ema_multi_avg_fn

# Setup based on the bug report: using sparse tensors
torch.manual_seed(42)

# Create sparse tensors
# Note: get_ema_multi_avg_fn expects parameters of the same shape for element-wise operations
indices = torch.tensor([[0, 1, 2], [0, 2, 3]])
values_A = torch.tensor([1.0, 2.0, 3.0])
values_B = torch.tensor([4.0, 5.0, 6.0])

A = torch.sparse_coo_tensor(indices, values_A, size=(3, 4))
B = torch.sparse_coo_tensor(indices, values_B, size=(3, 4))

# Adapt the call site to use the similar API
# Original: C = torch.sparse.mm(A, B)
# Similar: Use EMA update function
ema_avg_fn = swa_utils.get_ema_multi_avg_fn(decay=0.9)

# The function updates A in place
ema_avg_fn([A], [B], None)

# Verification: Check for corruption/crash on to_dense() as per the bug report
try:
    C = A.to_dense()
    print("Test passed: to_dense() succeeded without segmentation fault.")
except Exception as e:
    print(f"Test failed: {e}")