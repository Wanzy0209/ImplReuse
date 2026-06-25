import torch
import torch._inductor
# Test with TorchBench models and HF models to reproduce regressions
# Refer to dashboard for specific models showing performance degradation