import torch
import torch.nn.functional as F

# Configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True
torch.manual_seed(1352030645)

# Determine device (CUDA was used in the original bug report)
device = "cuda" if torch.cuda.is_available() else "cpu"

def test_nll_loss_divergence():
    # Adapted from the fuzzed_program style:
    # The original bug involved passing bfloat16 tensors where integer indices were expected.
    # For torch.nn.functional.nll_loss, the 'target' argument corresponds to indices.
    
    # Create input tensor (log-probabilities) with bfloat16
    # Shape (N, C) -> (4, 10)
    input_tensor = torch.full((4, 10), 0.5, dtype=torch.bfloat16, device=device)
    
    # Create target tensor (class indices) with bfloat16
    # Shape (N) -> (4,)
    # This mimics the problematic 'indices' tensor from the embedding bug.
    target_tensor = torch.full((4,), 1, dtype=torch.bfloat16, device=device)

    # Define the function to be tested/compiled
    def run_nll_loss(inp, tgt):
        return F.nll_loss(inp, tgt)

    # Attempt to compile and run to check for eager/compile divergence or assertions
    # The original bug triggered: assert "int" in str(indices.get_dtype())
    try:
        compiled_fn = torch.compile(run_nll_loss)
        result = compiled_fn(input_tensor, target_tensor)
        print("Test passed. Result:", result)
    except AssertionError as e:
        print(f"AssertionError caught (reproducing bug behavior): {e}")
    except RuntimeError as e:
        print(f"RuntimeError caught: {e}")

if __name__ == "__main__":
    test_nll_loss_divergence()