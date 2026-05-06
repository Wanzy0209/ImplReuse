import torch
import torch.nn as nn
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main():
    if not torch.cuda.is_available() or not torch.backends.cudnn.is_available():
        print("This bug requires a CUDA-enabled GPU with cuDNN.")
        return

    device = torch.device("cuda")
    dtype = torch.float32

    try:
        in_channels = 24
        
        model = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=(15, 3, 10),
            stride=(2, 1, 1),
            padding=(23, 0, 1),
            dilation=(1, 3, 3),
            groups=1,
            bias=False
        ).to(device, dtype=dtype)
        
        model.eval()

        input_shape = (1, in_channels, 24, 24, 24)
        input_tensor = torch.randn(input_shape, device=device, dtype=dtype)

        model(input_tensor)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()