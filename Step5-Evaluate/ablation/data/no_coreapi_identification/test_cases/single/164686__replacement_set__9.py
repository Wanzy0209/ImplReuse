import torch
import torch.nn.functional as F

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

# Replacement: torch.manual_seed -> torch.rand_like
# Original: torch.manual_seed(13653)
# New: Call torch.rand_like on an empty_strided tensor
_ = torch.rand_like(torch.empty_strided((), (), dtype=torch.float32))

def fuzzed_program(arg_0, arg_1, sentinel):
    # Replacement: torch.full -> torch.is_complex
    # Original: var_node_3 = torch.full((), 1.0, dtype=torch.float32)
    # New: Check if an empty_strided tensor is complex
    var_node_3 = torch.is_complex(torch.empty_strided((), (), dtype=torch.float32))
    
    var_node_2 = var_node_3.item() # dtype=float32 (bool -> float)
    
    var_node_5 = -3 # dtype=int32
    var_node_6 = arg_0 # dtype=int64
    var_node_4 = var_node_5 + var_node_6 # dtype=int32
    var_node_1 = var_node_2 + var_node_4 # dtype=float32
    
    var_node_9 = 1 # dtype=int64
    var_node_10 = -10 # dtype=int32
    var_node_8 = var_node_9 / var_node_10 # dtype=float64 (in python) or float (tensor)
    
    var_node_12 = arg_1 # dtype=int64
    var_node_13 = -5 # dtype=int32
    var_node_11 = var_node_12 / var_node_13 # dtype=float32
    
    var_node_7 = var_node_8 + var_node_11 # dtype=float32
    var_node_0 = var_node_1 * var_node_7 # dtype=float32
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Replacement: torch.tensor -> torch.empty_strided
# Original: sentinel = torch.tensor(1.0, requires_grad=True)
# New: Create empty_strided and fill it to ensure runnable behavior
sentinel = torch.empty_strided((), (), dtype=torch.float32, requires_grad=True)
sentinel.fill_(1.0)

# Helper to generate arguments using replacements
def get_int_arg():
    # Replacement: torch.tensor -> torch.empty_strided
    t = torch.empty_strided((), (), dtype=torch.int64)
    
    # Replacement: torch.randn -> torch.nn.functional.hinge_embedding_loss
    # We need inputs for hinge_embedding_loss. Use empty_strided for them too.
    inp = torch.empty_strided((), (), dtype=torch.float32)
    tgt = torch.empty_strided((), (), dtype=torch.float32)
    inp.fill_(1.0) # Initialize to avoid NaNs
    tgt.fill_(1.0)
    
    loss = F.hinge_embedding_loss(inp, tgt, margin=1.0, reduction='mean')
    t.copy_(loss.to(torch.int64))
    return t.item()

arg_0 = get_int_arg()
arg_1 = get_int_arg()

args = (arg_0, arg_1) + (sentinel,)
result_original = fuzzed_program(*args)
print(' eager success')
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print(' compile success')