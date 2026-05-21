import torch
import torch.backends.cuda

print(f"PyTorch Version: {torch.__version__}")

def run():
    """
    Test case for torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed.
    
    This test adapts the logic of the original bug report (Issue #162248), which 
    verified backend support for a specific operation (all_to_all in GLOO). 
    Here, we verify the backend capability reported by the similar API 
    (fp16/bf16 reduction in CUDA SDP).
    """
    # Setup: Check if the backend (CUDA) is available, analogous to 
    # initializing the process group in the original issue.
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        return

    # Action: Call the similar API to check the capability.
    # The original issue called dist.all_to_all which resulted in a RuntimeError.
    # We call this API to ensure it executes without error and returns a valid state.
    try:
        is_allowed = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
    except Exception as e:
        print(f"RuntimeError: {e}")
        raise

    # Verification: The original issue proved a feature was NOT supported (RuntimeError).
    # Here we verify the API returns a valid boolean indicating support status.
    assert isinstance(is_allowed, bool), \
        f"Expected boolean return type, got {type(is_allowed)}"

    print(f"fp16/bf16 reduction math SDP allowed: {is_allowed}")
    return is_allowed

if __name__ == "__main__":
    run()