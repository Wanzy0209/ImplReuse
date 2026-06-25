import torch
shape = (3,)
scale = 0.08547895223232808
zero_point = 70
data_fp = torch.tensor([-5.98352670669555664062500000000000e+00, -5.72708988189697265625000000000000e+00, -5.72708988189697265625000000000000e+00])
tensor_q1 = torch.quantize_per_tensor(data_fp, scale=scale, zero_point=zero_point, dtype=torch.quint2x4)
torch.equal(tensor_q1, tensor_q1)