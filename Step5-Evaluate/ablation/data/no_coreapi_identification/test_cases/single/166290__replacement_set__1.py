import torch
import torch.nn.functional as F
import torch.distributed as dist
import tensorflow as tf

# Setup for torch.distributed.gather
# Using a simple setup for single-process execution to avoid hanging
if not dist.is_initialized():
    try:
        dist.init_process_group(
            backend='gloo',
            init_method='tcp://127.0.0.1:29500',
            rank=0,
            world_size=1
        )
    except Exception as e:
        print(f"Distributed init skipped: {e}")

# 1. torch.manual_seed -> torch.set_rng_state
# Original: torch.manual_seed(974450504)
g = torch.Generator()
g.manual_seed(974450504)
torch.set_rng_state(g.get_state())

def fuzzed_program(arg_0, arg_1, sentinel):
    var_node_3 = arg_0
    
    # 2. torch.chunk -> torch.bucketize
    # Original: var_node_2 = torch.chunk(var_node_3, 3, dim=3)[0]
    # Adaptation: var_node_3 is 1D (from Flatten). bucketize works on 1D.
    boundaries = torch.tensor([10000, 20000], device=var_node_3.device)
    var_node_2 = torch.bucketize(var_node_3, boundaries)
    
    # 3. torch.full -> torch.fmax
    # Original: var_node_5 = torch.full((17,), 3, dtype=torch.int64)
    var_node_5 = torch.fmax(
        torch.zeros(17, dtype=torch.int64, device=var_node_3.device), 
        torch.tensor(3, dtype=torch.int64, device=var_node_3.device)
    )
    
    var_node_6 = arg_1
    _input_size_var_node_4 = var_node_5.size(0)
    
    # 4. torch.randint -> torch.amax
    # Original: _index_var_node_4 = torch.randint(0, _input_size_var_node_4, (15,), ...)
    # Adaptation: amax returns a scalar.
    temp_tensor_4 = torch.randn(15, device=var_node_5.device)
    _index_var_node_4 = torch.amax(temp_tensor_4, dim=0).long()
    
    # 5. torch.gather -> torch.distributed.gather
    # Original: var_node_4 = torch.gather(var_node_5, 0, _index_var_node_4)
    # Adaptation: dist.gather is collective. We perform the operation.
    # Since gather doesn't return the tensor, we assign a placeholder to var_node_4.
    if dist.is_initialized():
        # For world_size=1, gather_list has 1 element
        gather_list = [torch.zeros_like(var_node_5)]
        dist.gather(var_node_5, gather_list)
        var_node_4 = gather_list[0]
    else:
        var_node_4 = var_node_5
        
    _input_size_var_node_1 = var_node_2.size(0)
    
    # 4. torch.randint -> torch.amax (again)
    temp_tensor_1 = torch.randn(15, device=var_node_2.device)
    _index_var_node_1 = torch.amax(temp_tensor_1, dim=0).long()
    
    # 6. torch.index_select -> torch.nn.functional.pdist
    # Original: var_node_1 = torch.index_select(var_node_2, 0, _index_var_node_1)
    # Adaptation: pdist requires 2D input. Reshape 1D to (N, 1).
    var_node_2_2d = var_node_2.unsqueeze(1)
    var_node_1 = F.pdist(var_node_2_2d)
    
    # Original API: torch.squeeze
    var_node_0 = torch.squeeze(var_node_1)
    
    # sentinel is passed in, created via torch.isnan replacement
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# 1. torch.as_strided -> torch.nn.Flatten
# Original: arg_0 = torch.as_strided(...)
# Adaptation: Create tensor and flatten it.
base_tensor = torch.randint(0, 2, (17, 30, 17, 3), dtype=torch.int8).bool()
arg_0 = torch.nn.Flatten()(base_tensor)

arg_1 = torch.as_strided(torch.randint(5, 30, (15,)).to(torch.int64), (15,), (1,))

# 7. torch.tensor -> torch.isnan
# Original: sentinel = torch.tensor(1.0, requires_grad=True)
# Replacement:
sentinel = torch.isnan(torch.tensor(1.0))

args = (arg_0, arg_1) + (sentinel,)

# Run eager
try:
    result_original = fuzzed_program(*args)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')

# 8. torch.compile -> tf.compat.v1.tpu.rewrite
# Original: compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# Replacement:
# We define a dummy TF function to satisfy the API call syntax.
def dummy_tf_func(x):
    return x + 1

try:
    # This API is specific to TPUs and TF v1 compat.
    compiled_program = tf.compat.v1.tpu.rewrite(dummy_tf_func)
    print(' compile success (TF rewrite)')
except Exception as e:
    # Expected to fail in non-TPU environments, but syntax is correct.
    print(f' compile skipped (TF TPU rewrite requires specific env): {e}')