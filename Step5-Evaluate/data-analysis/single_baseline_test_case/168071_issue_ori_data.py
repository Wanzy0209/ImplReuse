# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch as pt
x0 = pt.zeros((6, 0))
x1 = pt.nn.functional.pad(
    x0, (0, 0, 0, 24)
)
print(x1.shape)