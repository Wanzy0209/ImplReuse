# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
print(torch.__version__,flush=True)
input = [
    [
        torch.empty((5, 7, 4, 3, 7, 6), dtype=torch.int8),
        torch.empty((4, 9, 2), dtype=torch.int32),
        (),
        False
    ],
    {},
    [],
    {}
]
torch.nn.functional.max_unpool1d(*input[0],**input[1])