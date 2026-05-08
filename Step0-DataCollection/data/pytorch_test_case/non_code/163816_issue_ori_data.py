device=torch.device('cuda')
in_features = 80
out_features = 128
dtype=torch.bfloat16
weight = torch.zeros(out_features, in_features, dtype=dtype, device=device)
bias = torch.zeros(out_features, dtype=dtype, device=device)
input = torch.zeros(2, 64, in_features, dtype=dtype, device=device)
print(torch.baddbmm(
    input=bias.view(1, 1, out_features),
    batch1=input,
    batch2=weight.mT.unsqueeze(0),
).shape)
Traceback (most recent call last):
  File "<string>", line 8, in <module>
RuntimeError: Expected size for first two dimensions of batch2 tensor to be: [2, 80] but got: [1, 80].