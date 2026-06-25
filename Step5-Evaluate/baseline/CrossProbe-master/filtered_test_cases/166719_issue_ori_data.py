import torch

class MyModule(torch.nn.Module):
    def __init__(self, param: int = 42) -> None:
        self.some_param = param  # Type checker error