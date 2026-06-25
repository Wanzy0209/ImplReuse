import torch
if torch.backends.mps.is_available():
    x = torch.tensor([1.0, 2.0, 3.0], device='mps')
    try:
        result = torch.special.gamma(x)
        print('gamma:', result)
    except Exception as e:
        print('gamma error:', e)
    try:
        result = torch.special.gammaincc(x, x)
        print('igammac:', result)
    except Exception as e:
        print('igammac error:', e)