# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
# repro_slice_huge_step_model.py
import os
import torch
from torch import nn

# Force CPU to minimize unrelated noise
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

class SliceHugeStepModel(nn.Module):
    def __init__(self, start=449, step=(2**63 - 1)):
        super().__init__()
        self.start = start
        self.step = step

    def forward(self, x: torch.Tensor):
        # Huge step makes the slice empty; eager returns shape (0,)
        sliced = torch.slice_copy(x, dim=0, start=self.start, end=None, step=self.step)
        return torch.reciprocal(sliced)

def get_input(n=875):
    return torch.randn(n, dtype=torch.float32)

def run_eager():
    m = SliceHugeStepModel()
    x = get_input()
    y = m(x)
    print("[eager] OK, output shape:", tuple(y.shape))

def run_compiled_aot_eager():
    m = SliceHugeStepModel()
    m_compiled = torch.compile(m, backend="aot_eager")
    x = get_input()
    y = m_compiled(x)
    print("[compile:aot_eager] OK, output shape:", tuple(y.shape))

def run_compiled_inductor_should_crash():
    print("[compile:inductor(default)] running ...")
    m = SliceHugeStepModel()
    m_compiled = torch.compile(m)  # default backend = inductor
    x = get_input()
    y = m_compiled(x)              # <-- segfault here
    print("[compile:inductor] (unexpected) survived, output shape:", tuple(y.shape))

if __name__ == "__main__":
    run_eager()
    run_compiled_aot_eager()
    run_compiled_inductor_should_crash()