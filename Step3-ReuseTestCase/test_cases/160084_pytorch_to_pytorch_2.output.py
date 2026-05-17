import torch
import torch.nn as nn

class ProdRegressionModel(torch.nn.Module):
    def __init__(self, a=1.0, b=0.0):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(a).float())
        self.b = torch.nn.Parameter(torch.tensor(b).float())
        self.first_batch = True

    def forward(self, x=None):
        if self.first_batch:
            print(f"Model dtype: {self.a.dtype}, {self.b.dtype}. Input dtype: {x.dtype}")
            self.first_batch = False
        
        # Adaptation: Using torch.prod instead of simple arithmetic
        # to test the similar API within the compilation context
        prod_x = torch.prod(x)
        return prod_x * self.a + self.b

def test_compile_with_prod():
    # The original bug is specific to CUDA, so we check for availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        return

    model = ProdRegressionModel()
    torch_device = "cuda"
    model.to(torch_device)
    
    # The regression occurs specifically with torch.compile and the inductor backend
    model.forward = torch.compile(model.forward, backend="inductor")
    
    inputs = torch.randn(4, 10).to(torch_device)
    
    try:
        output = model(inputs)
        assert isinstance(output, torch.Tensor)
        print("Test Passed: torch.compile with torch.prod executed successfully on CUDA.")
    except RuntimeError as e:
        print(f"Test Failed with RuntimeError: {e}")
        raise

if __name__ == "__main__":
    test_compile_with_prod()