# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
print(torch.__version__,flush=True)
value = -3.0397450923919677734375
shape = (2, 5, 1, 9)
scale = 0.10856233049200226
zero_point = 31
data_fp = torch.full(shape, value, dtype=torch.float32,device='cpu')
tensor_q = torch.quantize_per_tensor(data_fp, scale=scale, zero_point=zero_point, dtype=torch.quint2x4)
tensor_q = tensor_q.to('cuda')
input = [[tensor_q], {}, [], {}]
torch.Tensor.tolist(*input[0],**input[1])