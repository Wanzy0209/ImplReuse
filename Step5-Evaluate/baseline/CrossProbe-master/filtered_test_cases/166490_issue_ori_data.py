import torch
torch.manual_seed(42)
mat = torch.tensor([[1.,2.,3.,4.],[5.,6.,7.,8.],[9.,10.,11.,12.],[13.,14.,15.,16.]], dtype=torch.float32)
mat += 0.1
try:
    inv = torch.linalg.inv(mat)
except torch._C._LinAlgError as e:
    print('CPU Error:', e)
else:
    print('CPU Result:', inv)
if torch.cuda.is_available():
    mat_cuda = mat.cuda()
    inv_cuda = torch.linalg.inv(mat_cuda)
    print('CUDA Result:', inv_cuda)