# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
print(torch.__version__)
t = torch.zeros((5, 5, 9, 3), dtype=torch.complex32)
input = [
    [545460846592],
    {},
    [t],
    {}
]
r1 = torch.nn.PixelShuffle(*input[0],**input[1])
r2 = r1(*input[2],**input[3])