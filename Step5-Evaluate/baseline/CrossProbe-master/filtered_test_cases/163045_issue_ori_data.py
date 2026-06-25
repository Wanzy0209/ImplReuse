import torch

def test_graph_break_bomb_backend_inductor_device_xpu():
    if not torch.xpu.is_available():
        return
    device = 'xpu'
    @torch.compile(backend='inductor')
    def fn(x):
        return x + 1
    x = torch.randn(10, device=device)
    fn(x)

test_graph_break_bomb_backend_inductor_device_xpu()