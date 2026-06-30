import os
import torch
import torch.nn as nn

# Force CPU to minimize unrelated noise, consistent with the original bug report
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

class BatchNormEmptyInputModel(nn.Module):
    """
    Model using torch.nn.BatchNorm2d.
    This test case is derived from the issue where torch.slice_copy with a huge step
    (resulting in an empty tensor) caused a segfault in Inductor.
    We test BatchNorm with an empty tensor input to ensure similar Inductor stability.
    Note: SyncBatchNorm requires CUDA, so we use BatchNorm2d for CPU testing.
    """
    def __init__(self, num_features=3):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x: torch.Tensor):
        return self.bn(x)

def get_empty_input(batch_size=0, num_features=3, height=32, width=32):
    """
    Returns an empty tensor, mimicking the result of torch.slice_copy with a huge step.
    """
    return torch.randn(batch_size, num_features, height, width)

def run_eager():
    print("[eager] running ...")
    m = BatchNormEmptyInputModel()
    x = get_empty_input()
    y = m(x)
    print("[eager] OK, output shape:", tuple(y.shape))
    assert y.shape == (0, 3, 32, 32), "Eager execution produced unexpected shape"

def run_compiled_inductor():
    print("[compile:inductor] running ...")
    m = BatchNormEmptyInputModel()
    # Compile with default Inductor backend
    m_compiled = torch.compile(m)
    x = get_empty_input()
    
    # This should not segfault, similar to how aot_eager handled the slice_copy bug
    y = m_compiled(x)
    print("[compile:inductor] OK, output shape:", tuple(y.shape))
    assert y.shape == (0, 3, 32, 32), "Inductor produced unexpected shape"

if __name__ == "__main__":
    run_eager()
    run_compiled_inductor()