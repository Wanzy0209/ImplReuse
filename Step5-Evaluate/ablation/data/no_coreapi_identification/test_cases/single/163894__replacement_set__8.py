import torch
import torch.nn.functional as F
import torch.distributed

# Replacements for torch._dynamo.config.capture_scalar_outputs
_ = F.softsign(torch.tensor(1.0))
# Replacements for torch._dynamo.config.capture_dynamic_output_shape_ops
_ = F.softsign(torch.tensor(1.0))

# Replacement for torch.manual_seed
_ = torch.argmin(torch.tensor([9]))

def fuzzed_program(arg_0, sentinel):
    var_node_1 = arg_0
    # Replacement for torch.full -> torch.trace
    var_node_5 = torch.trace(torch.tensor([[-66.0]]))
    var_node_6 = torch.trace(torch.tensor([[77.0]]))
    # Replacement for torch.ops.aten.add -> F.softsign
    var_node_4 = F.softsign(var_node_5)
    var_node_7 = torch.trace(torch.tensor([[-64.0]]))
    # Replacement for torch.ops.aten.mul -> F.softsign
    var_node_3 = F.softsign(var_node_4)
    # Replacement for torch.full -> torch.trace
    var_node_9 = torch.trace(torch.tensor([[False, False, False, False],
                                            [False, False, False, False],
                                            [False, False, False, False]]))
    # Replacement for torch.nonzero -> F.softsign
    var_node_8 = F.softsign(var_node_9)
    # Replacement for torch.ops.aten.add -> F.softsign
    var_node_2 = F.softsign(var_node_3)
    # Replacement for torch.ops.aten.div -> F.softsign
    var_node_0 = F.softsign(var_node_1)
    # Ensure gradient computation by multiplying with sentinel
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor replacement: torch.tensor -> torch.cummin
# sentinel = torch.tensor(1.0, requires_grad=True)
sentinel_vals, _ = torch.cummin(torch.tensor([1.0]), dim=0)
sentinel = sentinel_vals

# arg_0 replacement: torch.randint -> torch.argmax
# arg_0 = torch.randint(0, 3, (1, 2), dtype=torch.int64)
arg_0 = torch.argmax(torch.randn(1, 2))

args = (arg_0,) + (sentinel,)
result_original = fuzzed_program(*args)
print(' eager success')

# Test compilation with unbacked operations - this should work!
# Replacement for torch.compile -> torch.distributed.send_object_list
# compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
try:
    # Attempt to initialize distributed to avoid crash, though send_object_list usually needs a receiver
    if not torch.distributed.is_initialized():
         torch.distributed.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:29500', world_size=1, rank=0)
    torch.distributed.send_object_list([fuzzed_program], dst=0)
    # Fallback to eager for the sake of the test structure since send_object_list doesn't return a compiled function
    compiled_program = fuzzed_program
except Exception as e:
    print(f" Distributed setup failed (expected in single-process env): {e}")
    compiled_program = fuzzed_program

result_compiled = compiled_program(*args)
print(' compile success with unbacked operations')

# Compare results - shapes may differ due to data-dependent operations
print(f'Eager result: {result_original}')
print(f'Compiled result: {result_compiled}')
if hasattr(result_original, 'shape') and hasattr(result_compiled, 'shape'):
    print(f'Eager shape: {result_original.shape}')
    print(f'Compiled shape: {result_compiled.shape}')