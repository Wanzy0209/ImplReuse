import torch
import torch.nn.functional as F

# Configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True
torch.manual_seed(1166094474)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, sentinel):
    # Original logic used torch.chunk to manipulate sizes.
    # Here we use torch.nn.functional.tanh (the similar API) on the inputs.
    # We preserve the logic of processing the inputs and combining them.

    # Apply tanh to the inputs
    # Note: tanh on bool/int inputs promotes to float
    out_0 = F.tanh(arg_0)
    out_1 = F.tanh(arg_1)
    out_2 = F.tanh(arg_2)
    out_3 = F.tanh(arg_3)

    # Combine results to mimic the structure of the original program
    # The original program concatenated chunks. Here we flatten and concatenate
    # to ensure we return a single tensor for the gradient check.
    res = torch.cat([
        out_0.flatten(),
        out_1.flatten(),
        out_2.flatten(),
        out_3.flatten()
    ])

    # Ensure gradient computation by multiplying with sentinel
    result = res * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

# Input generation from the original bug report
# These inputs use as_strided to create non-contiguous tensors, which is likely
# the source of the divergence in the original issue.
arg_0 = torch.as_strided(torch.randint(0, 2, (12,), dtype=torch.int8).bool(), (12,), (1,))
arg_1 = torch.as_strided(torch.randint(5, 30, (10,)).to(torch.int64), (10,), (1,))
arg_2 = torch.as_strided(torch.randint(0, 2, (24,), dtype=torch.int8).bool(), (6, 4), (4, 1))
arg_3 = torch.as_strided(torch.randint(0, 2, (2,), dtype=torch.int8).bool(), (2,), (1,))

args = (arg_0, arg_1, arg_2, arg_3) + (sentinel,)

# Test Eager
try:
    result_original = fuzzed_program(*args)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')
    raise

# Test Compiled
try:
    compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
    result_compiled = compiled_program(*args)
    print(' compile success')
except Exception as e:
    print(f' compile failed: {e}')
    raise

# Verify results match
assert torch.allclose(result_original, result_compiled), "Divergence between eager and compiled results"
print(' results match')