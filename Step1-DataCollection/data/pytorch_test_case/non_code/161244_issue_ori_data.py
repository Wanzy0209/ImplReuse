AssertionError: expected size 128==128, stride 1==64 at dim=1; expected size 64==64, stride 128==1 at dim=2
    Error in op: torch.ops.aten.polar.default
    This error most often comes from a incorrect fake (aka meta) kernel for a custom op.
    Use torch.library.opcheck to test your custom op.
    See https://pytorch.org/docs/stable/library.html#torch.library.opcheck