import torch
import torch.nn as nn
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main():
    if not torch.cuda.is_available():
        print("CUDA required")
        return

    device = torch.device("cuda")
    model = nn.ConvTranspose3d(
        in_channels=24,
        out_channels=1,
        kernel_size=(15, 3, 10),
        stride=(2, 1, 1),
        padding=(23, 0, 1),
        dilation=(1, 3, 3),
        bias=False
    ).to(device)
    
    input_tensor = torch.randn(1, 24, 24, 24, 24, device=device)
    model(input_tensor)

if __name__ == "__main__":
    main()