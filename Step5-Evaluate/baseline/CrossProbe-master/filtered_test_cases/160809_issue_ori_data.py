import torch
import time
import signal
import sys

def signal_handler(sig, frame):
    print('Signal received, cleaning up...')
    if torch.xpu.is_available():
        torch.xpu.synchronize()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if torch.xpu.is_available():
    device = torch.device('xpu')
    print(f'Using device: {device}')
    
    # Create tensors and perform operations that might trigger deadlock
    for i in range(100):
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.matmul(x, y)
        torch.xpu.synchronize()  # Force synchronization
        print(f'Iteration {i} completed')
        time.sleep(0.1)
else:
    print('XPU not available')