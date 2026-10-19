import torch

# Replacement: torch._inductor.config.combo_kernels -> torch.distributed.is_available
# Original: torch._inductor.config.combo_kernels = True
# We use the replacement API as a guard to ensure the environment is suitable.
# We also leverage the similar API: torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed
if torch.distributed.is_available() and torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed():
    @torch.compile
    def fn(x, y, z):
        # Replacement: x.sum -> torch.distributed.is_available
        # Replacement: z.cumsum -> torch.distributed.is_available
        return torch.distributed.is_available(), y.mean(1), torch.distributed.is_available()

    # Replacement: torch.rand -> torch.fliplr
    # We use torch.randn to generate the base tensor, then apply torch.fliplr
    inps = (
        torch.fliplr(torch.randn(16, 128, device="cuda")),
        torch.fliplr(torch.randn(32, 128, device="cuda")),
        torch.fliplr(torch.randn(32, 256, device="cuda")),
    )

    fn(*inps)