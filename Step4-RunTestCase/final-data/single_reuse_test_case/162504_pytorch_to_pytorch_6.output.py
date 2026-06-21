import torch
import torch.nn as nn
import sys

# Fix: Handle import compatibility for torch.export
# In PyTorch 2.1+, it is torch.export. In PyTorch 2.0, it was torch._export.
try:
    from torch.export import export
except ModuleNotFoundError:
    try:
        from torch._export import export
    except ModuleNotFoundError:
        print("torch.export module not found. This test requires PyTorch 2.1+ or PyTorch 2.0 with torch._export.")
        sys.exit(0)

# Define a module that uses torch.utils.checkpoint.checkpoint
class CheckpointedModel(nn.Module):
    def forward(self, x):
        def inner_fn(x):
            # Using torch.randn to introduce non-determinism that needs to be controlled via seed
            return x * torch.sigmoid(torch.randn(1, device=x.device))
        
        return torch.utils.checkpoint.checkpoint(
            inner_fn, x, 
            use_reentrant=False, 
            preserve_rng_state=True
        )

def main():
    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        return

    torch.cuda.manual_seed(42)

    # Initialize model and inputs
    model = CheckpointedModel().cuda()
    input_tensor = torch.ones(1, device="cuda", requires_grad=True)

    # 1. Eager Execution
    torch.cuda.manual_seed(42)
    eager_out = model(input_tensor)
    eager_grad, = torch.autograd.grad(eager_out, input_tensor)

    # 2. Export Execution
    # We use torch.export.export to trace the model containing the checkpoint
    torch.cuda.manual_seed(42)
    # Note: export() requires example inputs to trace the graph
    exported_model = export(model, (input_tensor,))
    
    torch.cuda.manual_seed(42)
    # Run the exported program
    export_out = exported_model(input_tensor)
    export_grad, = torch.autograd.grad(export_out, input_tensor)

    # 3. Verification
    # Check if outputs match
    assert torch.allclose(eager_out, export_out, rtol=0.0, atol=0.0), "Mismatch in outputs"
    # Check if gradients match
    assert torch.allclose(eager_grad, export_grad, rtol=0.0, atol=0.0), "Mismatch in gradient outputs"

    print("Eager Output:", eager_out)
    print("Export Output:", export_out)
    print("Eager Grad:", eager_grad)
    print("Export Grad:", export_grad)
    print("Test passed: torch.export.export works with torch.utils.checkpoint.checkpoint")

if __name__ == "__main__":
    main()