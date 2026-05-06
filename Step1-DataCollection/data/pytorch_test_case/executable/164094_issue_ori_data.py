class BackwardStream(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, stream: torch.cuda.Stream) -> torch.Tensor:
        ctx.stream = stream
        return input_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        stream = ctx.stream
        stream.wait_stream(torch.cuda.current_stream())
        torch.cuda.set_stream(stream)
        return grad_output