# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
# main.py
import torch
import os
from torch.utils.cpp_extension import load

op = load(name="add_extension", sources=[os.path.join(os.path.dirname(__file__), "op.cu")], verbose=True)

torch.library.define("myops::add_one", "(Tensor x) -> Tensor")
torch.library.define("myops::add_two", "(Tensor x) -> Tensor")
torch.library.impl("myops::add_one", "CUDA", op.add_one)
torch.library.impl("myops::add_two", "CUDA", op.add_two)

@torch.library.register_fake("myops::add_one")
def _(x): return torch.empty_like(x)

@torch.library.register_fake("myops::add_two")
def _(x): return torch.empty_like(x)

class M(torch.nn.Module):
    def forward(self, x):
        # return torch.ops.myops.add_one(x) works totally fine
        return torch.cond(x.shape[0] < 5, torch.ops.myops.add_one, torch.ops.myops.add_two, (x,))

model = M()

exported = torch.export.export(model, (torch.zeros(3, device="cuda"),), dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=128)}})
torch._inductor.aoti_compile_and_package(exported, package_path="model.pt2")
aoti_model = torch._inductor.aoti_load_package("model.pt2")
result = aoti_model(torch.zeros(6, device="cuda"))
print(f"{result = }")