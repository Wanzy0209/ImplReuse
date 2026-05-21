import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        # Adapted call site: torch.backends.cuda.cufft_plan_cache.size
        # Original API: torch.unique(x, sorted=True, return_inverse=True)
        # The similar API returns an integer, so we adapt the logic to convert it to a tensor.
        
        if x.is_cuda:
            device_idx = x.device.index
            # Call the similar API
            cache_size = torch.backends.cuda.cufft_plan_cache.size(device_idx)
        else:
            # Fallback for CPU to ensure the code structure remains valid
            cache_size = 0

        # Convert the integer result to a tensor to apply ReLU and return
        # This mimics the tensor flow of the original model
        t = torch.tensor(cache_size, device=x.device, dtype=x.dtype)
        return self.relu(t)

def main():
    print("torch.__version__ =", torch.__version__)

    model = MyModel()
    
    # Attempt to compile with fullgraph=True as in the original bug report
    # to see if the similar API triggers the same dynamic shape error.
    try:
        compiled_model = torch.compile(model, fullgraph=True)
    except Exception as e:
        print(f"Compilation setup failed: {e}")
        return

    # Create input
    x = torch.randn(8)
    if torch.cuda.is_available():
        x = x.cuda()
        model = model.cuda()
        compiled_model = compiled_model.cuda()

    try:
        model.eval()
        with torch.no_grad():
            y = compiled_model(x)
        print(f"[Test Passed] Output: {y}")
    except Exception as e:
        print(f"[Test Failed] Error: {e}")

if __name__ == "__main__":
    main()