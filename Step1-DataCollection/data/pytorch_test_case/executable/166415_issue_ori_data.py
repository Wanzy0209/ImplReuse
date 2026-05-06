import os
import torch
import torch_tensorrt

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch._inductor.aoti_load_package(os.path.join(os.getcwd(), "model.pt2"))
output = model(torch.randn(8, 10, device=device))
print(output)