import torch

# Fix: Check if torch._dynamo exists before accessing it to prevent AttributeError
has_dynamo = hasattr(torch, '_dynamo')
if has_dynamo:
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch.manual_seed(1014698)

def fuzzed_program(arg_0, sentinel):
    var_node_1 = arg_0 # size=(20, 0), stride=(1, 20), dtype=int64, device=cuda
    
    # Setup logic from the original fuzzer to maintain context
    var_node_3 = torch.full((), True, dtype=torch.bool) # size=(), stride=(), dtype=bool, device=cuda
    _x_nz = torch.zeros((), dtype=torch.bool, device=var_node_3.device)
    _x_nz_flat = _x_nz.reshape(-1)
    _x_nz_flat[:20] = True
    var_node_2 = torch.nonzero(_x_nz) # size=(20, 0), stride=(1, 20), dtype=int64, device=cuda
    
    # Adapted call site: replacing torch.add with torch.tanh
    # torch.tanh is a unary operation, so we apply it to the zero-sized tensor var_node_1
    var_node_0 = torch.tanh(var_node_1) # size=(20, 0), stride=(1, 20), dtype=int64, device=cuda
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

# Create the specific zero-sized tensor input
arg_0 = torch.as_strided(torch.randint(5, 30, (20,)).to(torch.int64), (20, 0), (1, 20))

args = (arg_0,) + (sentinel,)

# Initialize result_original to avoid potential NameError if eager mode fails
result_original = None

# Test Eager Mode
try:
    result_original = fuzzed_program(*args)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')

# Test Compiled Mode
if has_dynamo:
    try:
        compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
        result_compiled = compiled_program(*args)
        print(' compile success')
        
        # Verify consistency
        if result_original is not None and torch.equal(result_original, result_compiled):
            print(' results match')
        elif result_original is None:
            print(' results comparison skipped (eager mode failed)')
        else:
            print(' results mismatch')
    except Exception as e:
        print(f' compile failed: {e}')
else:
    print(' compile skipped: torch._dynamo not available')