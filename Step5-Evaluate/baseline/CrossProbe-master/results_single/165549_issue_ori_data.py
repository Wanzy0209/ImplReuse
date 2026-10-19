# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
# This fails - returns tensor with shape [0]
t = torch.randn(4, 4, device='privateuse1')
result = torch.abs(t)  # result.shape == torch.Size([0])

# But these work correctly
torch.abs(t, out=pre_allocated_tensor)  # Works
t.abs_()  # Works (in-place)
torch.neg(t)  # Works (uses structured_delegate)