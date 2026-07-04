import torch
from torch.distributions import constraints

# Configuration from the original bug report
# Check if _dynamo exists before accessing it to handle older PyTorch versions
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True

torch.manual_seed(974450504)

# Determine device (fallback to CPU if CUDA is not available to ensure runnability)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# --- Setup for the Similar API: torch.distributions.constraints.cat ---
# The original tensor var_node_1 has size (15, 30, 17, 1).
# We define a cat constraint that splits along dimension 0 into 3 chunks of size 5.
# We use boolean constraints as the original tensor was of dtype bool.
c1 = constraints.boolean
c2 = constraints.boolean
c3 = constraints.boolean
cat_constraint = constraints.cat([c1, c2, c3], dim=0, lengths=[5, 5, 5])

def fuzzed_program(arg_0, arg_1, sentinel):
    # Reproduce the tensor setup from the original bug report
    var_node_3 = arg_0  # size=(17, 30, 17, 3)
    var_node_2 = torch.chunk(var_node_3, 3, dim=3)[0]  # size=(17, 30, 17, 1)
    
    var_node_5 = torch.full((17,), 3, dtype=torch.int64, device=device)
    var_node_6 = arg_1
    
    _input_size_var_node_4 = var_node_5.size(0)
    _index_var_node_4 = torch.randint(0, _input_size_var_node_4, (15,), device=var_node_5.device)
    var_node_4 = torch.gather(var_node_5, 0, _index_var_node_4)
    
    _input_size_var_node_1 = var_node_2.size(0)
    _index_var_node_1 = torch.randint(0, _input_size_var_node_1, (15,), device=var_node_2.device)
    var_node_1 = torch.index_select(var_node_2, 0, _index_var_node_1) # size=(15, 30, 17, 1)

    # --- Adaptation: Replace torch.squeeze with torch.distributions.constraints.cat ---
    # Original: var_node_0 = torch.squeeze(var_node_1)
    # Similar API usage: We call the check method of the constraint.
    # Note: check returns a boolean tensor of the same shape as the input (or broadcastable),
    # indicating which elements satisfy the constraint.
    check_result = cat_constraint.check(var_node_1)
    
    # Adapt the result to fit the subsequent logic (multiplication by sentinel).
    # Since check_result is a boolean mask, we cast it to float to allow multiplication.
    var_node_0 = check_result.float()

    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True).to(device)

# Input setup
arg_0 = torch.as_strided(torch.randint(0, 2, (26010,), dtype=torch.int8).bool(), (17, 30, 17, 3), (1530, 51, 3, 1)).to(device)
arg_1 = torch.as_strided(torch.randint(5, 30, (15,)).to(torch.int64), (15,), (1,)).to(device)

args = (arg_0, arg_1, sentinel)

# Test Eager Execution
try:
    result_original = fuzzed_program(*args)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')

# Test Compiled Execution
# Check if torch.compile is available (requires PyTorch 2.0+)
if hasattr(torch, 'compile'):
    try:
        compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
        result_compiled = compiled_program(*args)
        print(' compile success')
    except Exception as e:
        print(f' compile failed: {e}')
else:
    print(' compile skipped: torch.compile not available in this PyTorch version')