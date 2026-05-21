import torch
import torch._dynamo

# Reproduce the configuration from the bug report
torch._dynamo.config.capture_scalar_outputs = True
torch.manual_seed(751735337)

# Check for CUDA availability as the original bug was device-specific
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, sentinel):
    # Adaptation: Create a valid input for torch.cholesky_inverse.
    # The API requires a square matrix representing a Cholesky factorization.
    # We construct a lower triangular matrix with positive diagonal values.
    
    # Create a random square matrix (e.g., 3x3)
    # Using float32 as cholesky_inverse does not support int16 used in the original gather test
    var_node_4 = torch.randn(3, 3, dtype=torch.float32, device=device)
    
    # Make it lower triangular
    var_node_3 = torch.tril(var_node_4)
    
    # Ensure the diagonal is positive to make it a valid Cholesky factor
    # We extract the diagonal, take absolute value, and place it back
    diag_indices = torch.arange(3, device=device)
    var_node_3[diag_indices, diag_indices] = torch.abs(var_node_3[diag_indices, diag_indices])
    
    # Call the similar API: torch.cholesky_inverse
    # This replaces the original torch.gather call site
    var_node_1 = torch.cholesky_inverse(var_node_3)
    
    return var_node_1

# Setup dummy arguments to match the function signature
# The shapes are arbitrary here as we generate the critical tensor inside,
# but we maintain the signature to mimic the fuzzer harness.
args = [torch.randn(10, 10, device=device) for _ in range(7)]

def run_test():
    try:
        # Run in eager mode
        eager_result = fuzzed_program(*args, None)
        
        # Run in compiled mode (torch._dynamo)
        # The bug report indicates an issue with guards/compilation
        compiled_fn = torch.compile(fuzzed_program)
        compiled_result = compiled_fn(*args, None)
        
        # Check for eager/compile divergence
        if not torch.allclose(eager_result, compiled_result):
            print("Test Failed: Eager and Compiled results diverged.")
            return False
        else:
            print("Test Passed: Eager and Compiled results match.")
            return True
            
    except AssertionError as e:
        # Check for the specific assertion mentioned in the bug report
        if "guard_or_defer_runtime_assert" in str(e):
            print(f"Bug Reproduced: {e}")
            return False
        else:
            raise

if __name__ == "__main__":
    run_test()