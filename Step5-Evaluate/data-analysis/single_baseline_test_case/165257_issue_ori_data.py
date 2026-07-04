# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch

# Test each operation on CPU and MPS
ops = [
    ("normal_(0,1)", lambda t: t.normal_(0, 1)),
    ("uniform_(0,1)", lambda t: t.uniform_(0, 1)),
    ("exponential_(1)", lambda t: t.exponential_(1.0)),
    ("bernoulli_(0.5)", lambda t: t.bernoulli_(0.5)),
    ("random_()", lambda t: t.random_()),
    ("random_(10)", lambda t: t.random_(10)),
    ("random_(0,10)", lambda t: t.random_(0, 10)),
]

print(f"{'Operation':<20} {'CPU Max':<12} {'MPS Max':<12} {'Status'}")
print("-" * 60)

for name, op_func in ops:
    # CPU: works correctly
    t_cpu = torch.zeros(50, 50, device='cpu').T.clone()  # Non-contiguous
    op_func(t_cpu)
    cpu_max = t_cpu.max().item()

    # MPS: fails silently
    t_mps = torch.zeros(50, 50, device='mps').T.clone()  # Non-contiguous
    op_func(t_mps)
    mps_max = t_mps.max().item()

    status = "✓ OK" if mps_max != 0.0 else "✗ BUG"
    print(f"{name:<20} {cpu_max:<12.4f} {mps_max:<12.4f} {status}")