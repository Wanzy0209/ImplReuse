import torch
from torch import nn

class EqualModule(nn.Module):
    def __init__(
        self,
        reference: torch.Tensor,
    ) -> None:
        super().__init__()
        # Store the reference tensor to compare against
        self.ref = nn.Parameter(reference, requires_grad=False)

    def forward(self, x: torch.Tensor) -> bool:
        # Replace torch.searchsorted with torch.equal
        return torch.equal(self.ref, x)

def test_device(device, x, reference):
    # Check if torch.compile is available (requires PyTorch 2.0+)
    if not hasattr(torch, 'compile'):
        print(f"Skipping test on {device}: torch.compile is not available in this PyTorch version.")
        return

    x = x.to(device)
    reference = reference.to(device)
    
    module = EqualModule(reference).to(device)
    module_compiled = torch.compile(EqualModule(reference).to(device), fullgraph=True)

    # warm up
    with torch.no_grad():
        y_original = module(x)
        y_compiled = module_compiled(x)

    # proper inference
    with torch.no_grad():
        y_original = module(x)
        y_compiled = module_compiled(x)

    print(f'device: {device}, original: {y_original}, compiled: {y_compiled}')
    
    # Assert that the compiled version produces the same boolean result as the original
    assert y_original == y_compiled, (
        f"torch.equal mismatch on {device}: "
        f"original={y_original}, compiled={y_compiled}"
    )

def main():
    batch_size = 32
    feature_dim = 10
    torch.manual_seed(42)

    # Test Case 1: Tensors are equal (Expected result: True)
    x_equal = torch.randn(batch_size, feature_dim, dtype=torch.float32)
    ref_equal = x_equal.clone()
    
    print("Testing with equal tensors:")
    test_device('cpu', x_equal, ref_equal)
    if torch.cuda.is_available():
        test_device('cuda', x_equal, ref_equal)

    # Test Case 2: Tensors are not equal (Expected result: False)
    x_unequal = torch.randn(batch_size, feature_dim, dtype=torch.float32)
    ref_unequal = torch.randn(batch_size, feature_dim, dtype=torch.float32)
    
    print("\nTesting with unequal tensors:")
    test_device('cpu', x_unequal, ref_unequal)
    if torch.cuda.is_available():
        test_device('cuda', x_unequal, ref_unequal)

if __name__ == '__main__':
    main()