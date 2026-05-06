import torch
import torch.nn as nn
from tensordict import NonTensorData, TensorDict
from tensordict.nn import dispatch


def test(cls: type[nn.Module]) -> None:
    instance = cls()
    tensordict = TensorDict(
        {"a": torch.zeros(3, 5), "b": torch.ones(3, 2)}, batch_size=(3,)
    )

    print("Mode 1")
    print("legacy\n", instance(tensordict["a"], tensordict["b"]))
    print("up to date\n", instance(tensordict))

    print("Mode 2")
    instance.mode = "mode2"
    print("legacy\n", instance(tensordict["a"], tensordict["b"]))
    print("up to date\n", instance(tensordict))