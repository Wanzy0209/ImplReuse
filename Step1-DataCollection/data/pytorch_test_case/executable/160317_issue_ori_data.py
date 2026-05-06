import torch

class ReluOps(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x.relu_()
        ctx.mark_dirty(x)
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad):
        x = ctx.saved_tensors
        return (grad * x[0])

def run():
    for i in range(100):
        x = torch.rand((100, 100, 1000), requires_grad=True)
        z = x + 1.0
        z_view = z[0]
        ReluOps.apply(z_view)

if __name__ == "__main__":
    run()