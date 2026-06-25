import torch
if torch.xpu.is_available():
    try:
        print(torch.xpu.mem_get_info())
    except RuntimeError as e:
        print(f'Error: {e}')