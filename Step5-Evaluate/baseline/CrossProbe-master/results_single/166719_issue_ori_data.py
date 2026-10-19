# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch

class MyModule(torch.nn.Module):
    def __init__(self, param: int = 42) -> None:
        self.some_param = param  # <- raises an `unresolved-attribute` error