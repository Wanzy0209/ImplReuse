import torch

torch.manual_seed(1337)

@torch.compile()
def vec_norm(e_dist):
    return torch.nn.functional.normalize(e_dist)

def vec_norm_without_compile(e_dist):
    return torch.nn.functional.normalize(e_dist)

device='cuda'
c=torch.tensor([[3.799999 ,0.0, 0.0]],device=device,dtype=torch.float32)
print("Input vector",[x.item() for x in c[0] ])
xyz=vec_norm(c)
print("Normalized vector (compile):",[x.item() for x in xyz[0] ],"torch.acos of component 0",torch.acos(xyz[0,0]).item())
xyz=vec_norm_without_compile(c)
print("Normalized vector (without compile):",[x.item() for x in xyz[0] ],"torch.acos of component 0",torch.acos(xyz[0,0]).item())