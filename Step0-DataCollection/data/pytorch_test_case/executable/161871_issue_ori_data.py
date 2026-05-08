import torch
print(torch.__version__)

tensor = tensor = torch.randint(low=0, high=10, size=(5,),dtype=torch.int64)

input = [[tensor,1024],{}]

torch.Tensor.mvlgamma_(*input[0],**input[1])