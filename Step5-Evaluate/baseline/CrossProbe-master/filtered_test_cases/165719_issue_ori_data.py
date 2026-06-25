import torch

class TestTritonDotReduction:
    def test_3mm_add(self):
        # Simplified reproduction attempt
        a = torch.randn(10, 10, device='xpu')
        b = torch.randn(10, 10, device='xpu')
        c = torch.randn(10, 10, device='xpu')
        result = torch.addmm(a, b, c)
        return result