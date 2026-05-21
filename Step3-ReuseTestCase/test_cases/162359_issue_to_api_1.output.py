import torch
from torch import optim

def test_sequentiallr_bug_with_cat():
    """
    Reproduces the SequentialLR bug where using a Tensor learning rate
    causes base_lrs to be corrupted due to aliasing.
    Leverages torch.cat to construct the learning rate tensor.
    """
    # 1. Use a tensor learning rate constructed via torch.cat
    # This leverages the similar API as requested.
    # We construct a 1D tensor and squeeze it to a scalar tensor.
    lr_tensor = torch.cat([torch.tensor([1
    assert lr_tensor
