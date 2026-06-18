import torch
import torch.nn as nn

class MyOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor):
        out_0 = torch.zeros(2**20, device=inp.device, dtype=torch.float32)
        out_1 = torch.zeros(2**20, device=inp.device, dtype=torch.float32)
        ctx.save_for_backward(
            inp,
            out_0,
            out_1,
        )
        return out_0, out_1

    @staticmethod
    def backward(ctx, dA, dB):
        _ = ctx.saved_tensors  # this is necessary
        return None

class MyModel(nn.Module):
    def forward(self, x):
        # Apply the custom autograd function
        return MyOp.apply(x)[0]

def test_dataparallel_memory_leak():
    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel().to(device)
    
    if torch.cuda.is_available():
        # Wrap the model with DataParallel
        # We use device_ids=[0] to ensure it runs even with a single GPU
        model = nn.DataParallel(model, device_ids=[0])
        
        dummy_input = torch.nn.Parameter(torch.randn(2**20, device=device))
        
        # Run iterations to check for memory leaks
        for i in range(100):
            full_out = model(dummy_input)
            full_out.sum().backward()
            dummy_input.grad = None  # free gradient memory
            print(i, torch.cuda.memory_allocated() / 1024**2, "MiB")
    else:
        print("CUDA not available, skipping test.")

if __name__ == "__main__":
    test_dataparallel_memory_leak()