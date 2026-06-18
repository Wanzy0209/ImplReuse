import torch
import torch.nn as nn

class ProdModel(nn.Module):
    """
    A simple model adapted to use torch.prod to test for numerical consistency,
    similar to the original issue reported with torch.compile on a Transformer.
    """
    def __init__(self, input_size, output_size):
        super(ProdModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.linear(x)
        # Using torch.prod as the core operation to verify numerical stability
        # This replaces the complex attention mechanism from the original test case
        return torch.prod(x, dim=-1)

def test_torch_prod_consistency():
    # Setup parameters
    batch_size = 2
    seq_length = 5
    input_size = 10
    output_size = 10

    # Initialize model and input
    model = ProdModel(input_size, output_size)
    input_tensor = torch.randn(batch_size, seq_length, input_size)

    # 1. Run in eager mode
    output_eager = model(input_tensor)

    # 2. Run in compiled mode (context of the original bug)
    compiled_model = torch.compile(model)
    output_compiled = compiled_model(input_tensor)

    # 3. Verify numerical consistency
    # The original bug reported "severe" inconsistencies. We check if the outputs
    # are close within a reasonable tolerance for float32 operations.
    assert torch.allclose(output_eager, output_compiled, rtol=1e-4, atol=1e-5), \
        f"Numerical inconsistency detected between eager and compiled modes. Max diff: {torch.max(torch.abs(output_eager - output_compiled))}"

    print("Test passed: torch.prod is numerically consistent with torch.compile.")

if __name__ == "__main__":
    test_torch_prod_consistency()