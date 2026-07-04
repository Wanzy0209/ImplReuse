import torch
import torch.nn.functional as F

# Setup from the original issue
# Handle cases where torch._dynamo might not be available (e.g., older versions or specific builds)
try:
    torch._dynamo.config.capture_scalar_outputs = True
except AttributeError:
    pass

torch.manual_seed(1352030645)

# Determine device (original was cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fuzzed_program(arg_0, arg_1, sentinel):
    # Adapted logic to test torch.nn.functional.cross_entropy
    # The original bug involved passing float (bfloat16) tensors as indices to embedding.
    # Here we test passing bfloat16 tensors as targets to cross_entropy.
    
    # Input logits: (Batch=4, Classes=10)
    # Using bfloat16 to match the fuzzer environment
    input_tensor = torch.randn(4, 10, dtype=torch.bfloat16, device=device)
    
    # Target indices: (Batch=4)
    # Intentionally using bfloat16 for target (indices) to mimic the bug scenario
    # where embedding received float indices. This might trigger a divergence
    # or assertion error similar to the original issue.
    target_tensor = torch.randint(0, 10, (4,), dtype=torch.bfloat16, device=device)

    # Call the similar API
    return F.cross_entropy(input_tensor, target_tensor)

if __name__ == "__main__":
    # Check for bfloat16 support
    if device.type == "cuda" or (device.type == "cpu" and torch.cpu.is_avx512_supported()):
        # Create dummy arguments to match the signature structure
        arg_0 = torch.empty(1, device=device)
        arg_1 = torch.empty(1, device=device)
        
        print("Testing Eager mode...")
        try:
            out_eager = fuzzed_program(arg_0, arg_1, None)
            print(f"Eager output: {out_eager}")
        except Exception as e:
            print(f"Eager Exception: {e}")

        print("\nTesting Compiled mode...")
        try:
            compiled_fn = torch.compile(fuzzed_program)
            out_compiled = compiled_fn(arg_0, arg_1, None)
            print(f"Compiled output: {out_compiled}")
        except Exception as e:
            print(f"Compiled Exception: {e}")
    else:
        print("Test skipped: bfloat16 not supported on this device.")