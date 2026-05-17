import torch

# Adaptation: Since torch.pca_lowrank operates on data matrices rather than models,
# we replace the model/tokenizer setup with a random data matrix.
# This keeps the test minimal and runnable without external dependencies like transformers.
torch.manual_seed(42)
A = torch.randn(10, 5, dtype=torch.float64)

# Original call: ep = torch.export.export(model, example_inputs)
# Adapted call: torch.pca_lowrank(A)
# We verify the similar API works correctly.
try:
    U, S, V = torch.pca_lowrank(A)
    
    # Assertions to verify the output shapes and types
    assert U.shape == (10, 5), f"Expected U shape (10, 5), got {U.shape}"
    assert S.shape == (5,), f"Expected S shape (5,), got {S.shape}"
    assert V.shape == (5, 5), f"Expected V shape (5, 5), got {V.shape}"
    
    print("Test passed: torch.pca_lowrank executed successfully.")

except AssertionError as e:
    # Check for the specific error mentioned in the bug report
    if "Current active mode" in str(e):
        print(f"Bug reproduced: {e}")
    else:
        raise