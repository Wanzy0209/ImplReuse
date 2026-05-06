>>> import torch
>>> torch.constant_pad_nd(torch.ones([2, 3], device="cpu"), [])
tensor([[1., 1., 1.],
        [1., 1., 1.]])
>>> torch.constant_pad_nd(torch.ones([2, 3], device="mps"), [])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: invalid padding argument of size 0