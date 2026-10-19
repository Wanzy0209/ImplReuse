import torch
import torch.nn.functional as F

# Test case for Issue 166888: NameError when using .item() on float tensor arg in torch.compile
# Adapted to use Similar API: torch.nn.functional.tanh
# Replacements applied:
# - torch.randn -> torch.nn.functional.max_pool2d
# - torch.tensor -> torch.cummax

def test_inductor_item_with_tanh():
    # Setup x: Replace torch.randn with torch.nn.functional.max_pool2d
    # We create a dummy input for max_pool2d using torch.ones to avoid using torch.randn
    dummy_input = torch.ones(10, 1, 20, 30, device='cuda')
    x = F.max_pool2d(dummy_input, kernel_size=1).squeeze(1)

    # Setup max_val: Replace torch.tensor with torch.cummax
    # We create a dummy input for cummax using torch.arange to avoid using torch.tensor
    # torch.arange(1.0, 6.0, 4.0) produces [1.0, 5.0]
    dummy_vals = torch.arange(1.0, 6.0, 4.0, device='cuda')
    max_val, _ = torch.cummax(dummy_vals, dim=0)
    max_val = max_val[-1] # Extract the 5.0

    # Function using Similar API (tanh) and Bug Logic (.item())
    def f(x, scale_val):
        # Using tanh instead of clamp to test the similar API
        y = F.tanh(x * scale_val.item())
        return y

    # Compile
    compiled_func = torch.compile(f, backend='inductor', fullgraph=True)

    # Run
    try:
        result = compiled_func(x, max_val)
        print("Test passed. Result shape:", result.shape)
        assert result.shape == x.shape
    except Exception as e:
        print("Test failed with error:", e)
        raise

if __name__ == "__main__":
    test_inductor_item_with_tanh()