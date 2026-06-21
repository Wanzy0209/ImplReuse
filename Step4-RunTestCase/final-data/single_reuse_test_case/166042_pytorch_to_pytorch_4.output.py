import torch
import torch.nn.functional as F

# Configuration from the original bug report
# Check if _dynamo exists to avoid AttributeError in older PyTorch versions
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True

torch.manual_seed(1352030645)

def fuzzed_program_kl_div(arg_0, arg_1):
    # Context: bfloat16, cuda (mimicking the original fuzzer environment)
    # arg_0: size=(4, 8), dtype=bfloat16, device=cuda
    # arg_1: size=(8, 7), dtype=bfloat16, device=cuda
    
    # Pre-processing operations similar to the original fuzzer output
    var_node_5 = torch.full((8, 7), -0.80078125, dtype=torch.bfloat16, device=arg_0.device)
    var_node_3 = torch.matmul(arg_0, var_node_5)
    
    # Prepare inputs for torch.nn.functional.kl_div
    # input: log-probabilities (using the matmul result)
    # target: probabilities (generating a tensor of same shape)
    kl_input = var_node_3
    kl_target = torch.full((4, 7), 0.5, dtype=torch.bfloat16, device=arg_0.device)
    
    # Call the similar API: torch.nn.functional.kl_div
    # Note: reduction='mean' is used as a default
    output = F.kl_div(kl_input, kl_target, reduction='mean')
    
    return output

if torch.cuda.is_available():
    # Initialize inputs with specific dtypes and devices
    arg_0 = torch.randn(4, 8, dtype=torch.bfloat16, device='cuda')
    arg_1 = torch.randn(8, 7, dtype=torch.bfloat16, device='cuda')
    
    # Check if torch.compile is available (requires PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        # Run with torch.compile to check for eager/compile divergence
        # This mirrors the scenario in the original bug report
        try:
            compiled_fn = torch.compile(fuzzed_program_kl_div)
            result = compiled_fn(arg_0, arg_1)
            print("Test passed: torch.nn.functional.kl_div executed successfully under torch.compile.")
        except Exception as e:
            print(f"Test failed with error: {e}")
    else:
        print("torch.compile is not available (requires PyTorch 2.0+). Skipping test.")
else:
    print("CUDA device not found. Skipping test.")