import torch
import torch.nn.functional as F

def test_embedding_indices_dtype_assertion():
    """
    Test case for Issue 166042: [Fuzzer][Eager/Compile Divergence] 
    assert "int" in str(indices.get_dtype())
    
    This test reproduces the scenario where torch.compile (torch._dynamo) 
    encounters an embedding operation with indices that are not of integer type 
    (e.g., bfloat16), triggering an assertion error in the backend.
    """
    # Configuration from the bug report
    torch._dynamo.config.capture_scalar_outputs = True
    
    # Define a function that performs embedding lookup
    # The bug is triggered when the indices tensor has a non-integer dtype.
    def embedding_model(weight, indices):
        return F.embedding(indices, weight)

    # Setup inputs
    # Create a weight tensor (standard float)
    weight = torch.randn(10, 5, dtype=torch.float32)
    
    # Create indices with bfloat16 dtype to trigger the assertion.
    # The fuzzer generated bfloat16 tensors which were eventually used as indices.
    # The assertion "assert 'int' in str(indices.get_dtype())" expects the string 
    # representation of the dtype to contain 'int'.
    indices_bf16 = torch.tensor([0, 1, 2], dtype=torch.bfloat16)

    # Compile the function using torch.compile (torch._dynamo)
    try:
        compiled_model = torch.compile(embedding_model)
        result = compiled_model(weight, indices_bf16)
        print("Test passed (Bug might be fixed or path not taken).")
    except AssertionError as e:
        # Check for the specific assertion mentioned in the issue title
        if "int" in str(e) and "dtype" in str(e):
            print(f"Bug reproduced: {e}")
            return True
        else:
            raise
    except Exception as e:
        # Depending on the PyTorch version, this might raise a TypeError before compilation
        # or a different error during compilation.
        print(f"Exception raised during execution: {type(e).__name__}: {e}")
        return False

    return False

if __name__ == "__main__":
    test_embedding_indices_dtype_assertion()