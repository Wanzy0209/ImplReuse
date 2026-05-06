raise e.with_traceback(None) from e.__cause__  # User compiler error
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch._dynamo.exc.Unsupported: Skip calling `torch.compiler.disable()`d function