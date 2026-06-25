import torch

class TestOpenReg:
    def test_copy_same_device(self):
        # Simplified reproduction attempt
        x = torch.randn(3, 3)
        y = x.copy_(x)  # Potential failing operation
        assert torch.equal(x, y)