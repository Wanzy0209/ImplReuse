import torch
class NoMixOrderReductionTest:
    def test_rms_norm_bwd_float32_shape1(self):
        x = torch.randn(1, 1024, dtype=torch.float32, device='xpu')
        x.requires_grad_(True)
        y = torch.nn.functional.rms_norm(x, (1024,))
        loss = y.sum()
        loss.backward()
