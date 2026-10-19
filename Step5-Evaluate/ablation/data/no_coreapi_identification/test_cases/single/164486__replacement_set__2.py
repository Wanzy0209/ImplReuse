import torch
import tensorflow as tf
import torch.distributed as dist

# Setup for distributed to make it runnable (best effort)
# Note: torch.distributed.reduce requires an initialized process group.
try:
    if not dist.is_initialized():
        # Using a simple TCP store for single-process testing if possible, 
        # or just gloo backend which might work in some restricted envs.
        # This often requires a running rendezvous server, so we wrap it.
        dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:12345', rank=0, world_size=1)
except Exception as e:
    print(f"Distributed init skipped/failed (expected in some envs): {e}")

torch._dynamo.config.capture_scalar_outputs = True

# Replacement: torch.manual_seed -> tf.compat.v1.assign
# Original: torch.manual_seed(238)
# Replacement: tf.compat.v1.assign(tf.Variable(238), 238)
tf.compat.v1.assign(tf.Variable(238), 238)

def fuzzed_program(arg_0, sentinel):
    # Replacement: torch.full -> torch.fmin
    # Original: var_node_2 = torch.full((), 1, dtype=torch.int16)
    # Replacement: torch.fmin(torch.tensor(1, dtype=torch.int16), torch.tensor(1, dtype=torch.int16))
    var_node_2 = torch.fmin(torch.tensor(1, dtype=torch.int16), torch.tensor(1, dtype=torch.int16))
    
    var_node_3 = arg_0
    
    # Replacement: torch.add -> torch.sub
    # Original: var_node_1 = torch.add(var_node_2, var_node_3)
    # Replacement: var_node_1 = torch.sub(var_node_2, var_node_3)
    var_node_1 = torch.sub(var_node_2, var_node_3)
    
    # Replacement: torch.full -> torch.fmin
    # Original: var_node_5 = torch.full((1,), 3, dtype=torch.int16)
    # Replacement: torch.fmin(torch.tensor([3], dtype=torch.int16), torch.tensor([3], dtype=torch.int16))
    var_node_5 = torch.fmin(torch.tensor([3], dtype=torch.int16), torch.tensor([3], dtype=torch.int16))
    
    # Replacement: torch.squeeze -> torch.nn.LazyLinear
    # Original: var_node_4 = torch.squeeze(var_node_5)
    # Replacement: torch.nn.LazyLinear(1)(var_node_5)
    # LazyLinear is a module, we instantiate and call it. 
    # Input (1,) is treated as (1, 1) by Linear, output is (1, 1) -> squeeze not needed if we want (1,) but LazyLinear returns (batch, out_features).
    # To match original shape (1,), we might need to squeeze, but we are replacing squeeze.
    # Let's assume the output of LazyLinear(1) on (1,) is (1, 1). 
    # The original var_node_4 was size (). 
    # This is a semantic shift, but we follow the replacement.
    # To keep it runnable and compatible with subsequent ops (which expect scalar or specific shapes), 
    # we note that var_node_4 was used in torch.div(var_node_1, var_node_4).
    # var_node_1 is (). var_node_4 is ().
    # If var_node_4 becomes (1, 1), div might broadcast.
    # Let's try to keep shapes compatible if possible, or just apply the replacement.
    # torch.nn.LazyLinear(1) returns a module. Calling it on (1,) returns (1, 1).
    # We will just apply the replacement.
    var_node_4 = torch.nn.LazyLinear(1)(var_node_5)
    
    var_node_0 = torch.div(var_node_1, var_node_4)
    
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Replacement: torch.tensor -> torch.isinf
# Original: sentinel = torch.tensor(1.0, requires_grad=True)
# Replacement: sentinel = torch.isinf(torch.tensor(1.0))
sentinel = torch.isinf(torch.tensor(1.0))

# Replacement: torch.randn -> tf.random.stateless_truncated_normal
# Original: arg_0 = torch.as_strided(torch.randn(1).to(torch.int16), (), ())
# Replacement: arg_0 = torch.as_strided(tf...view...
# We replace torch.randn with the TF equivalent, convert to PT, then apply as_strided -> view replacement.
tf_tensor = tf.random.stateless_truncated_normal([1], [0, 0], dtype=tf.float32)
torch_tensor = torch.from_numpy(tf_tensor.numpy()).to(torch.int16)
# Replacement: torch.as_strided -> torch.Tensor.view
# Original: torch.as_strided(..., (), ())
# Replacement: ... .view(())
arg_0 = torch_tensor.view(())

args = (arg_0,) + (sentinel,)
out_eager = fuzzed_program(*args)
out_eager.sum().backward()
print('Eager Success! ')

# Replacement: torch.compile -> torch.distributed.reduce
# Original: compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# Replacement: compiled_program = torch.distributed.reduce(...)
# torch.distributed.reduce is a collective op, not a compiler wrapper. 
# It returns None. We perform the operation on a dummy tensor to satisfy the call.
compiled_program = None
try:
    # Passing a dummy tensor to avoid modifying actual data
    torch.distributed.reduce(torch.tensor(0.0), 0)
except Exception as e:
    print(f"Distributed reduce failed: {e}")

if compiled_program is not None:
    out_compiled = compiled_program(*args)
    out_compiled.sum().backward()
    print('Compile Success! ')
    out_eager_sum = out_eager.sum()
    out_compiled_sum = out_compiled.sum()
    diff = (out_eager_sum - out_compiled_sum).abs().item()
    rel_diff = diff / (out_eager_sum.abs().item() + 1e-12) * 100
    print(f'Relative diff (sum): {rel_diff:.6f}%')
    if rel_diff > 5 and diff > 1:
        print(f' Forward output sums differ significantly (relative and absolute)!')
        print('out_eager_sum:', out_eager_sum.item())
        print('out_compiled_sum:', out_compiled_sum.item())
        print('Absolute diff:', diff)
        print('Relative diff (%):', rel_diff)
        import sys; sys.exit(1)
else:
    print("Skipping compiled execution as torch.compile was replaced with torch.distributed.reduce.")
    assert result
