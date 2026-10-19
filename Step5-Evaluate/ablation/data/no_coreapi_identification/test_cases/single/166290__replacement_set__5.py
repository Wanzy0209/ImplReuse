import torch
import torch.nn.functional as F
import tensorflow as tf

# torch.manual_seed -> torch.isinf
torch.isinf(torch.tensor(0.0))

def fuzzed_program(arg_0, arg_1, sentinel):
    var_node_3 = arg_0
    # torch.chunk -> torch.nn.functional.pdist
    # var_node_3 is (17, 30, 3). pdist expects 2D.
    # Reshape to (1, N) to make it work.
    var_node_2 = F.pdist(var_node_3.flatten().unsqueeze(0))

    # torch.full -> torch.amax
    var_node_5 = torch.amax(var_node_3, dim=0)

    # torch.randint -> torch.full
    _input_size_var_node_4 = var_node_5.size(0)
    _index_var_node_4 = torch.full((15,), 0, dtype=torch.int64, device=var_node_5.device)

    # torch.gather -> torch.round
    var_node_4 = torch.round(var_node_5)

    # torch.randint -> torch.full
    _input_size_var_node_1 = var_node_2.size(0)
    _index_var_node_1 = torch.full((15,), 0, dtype=torch.int64, device=var_node_2.device)

    # torch.index_select -> torch.cummax
    var_node_1 = torch.cummax(var_node_2, dim=0)[0]

    # torch.squeeze (Target API)
    var_node_0 = torch.squeeze(var_node_1)

    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# torch.tensor -> tf.keras.ops.diag
sentinel = torch.from_numpy(tf.keras.ops.diag([1.0]).numpy()).float().requires_grad_(True)

# torch.as_strided -> torch.bmm
# Original arg_0 was (17, 30, 17, 3).
# bmm input: (17, 30, 17) and (17, 17, 3) -> (17, 30, 3)
t1 = torch.randint(0, 2, (17, 30, 17))
t2 = torch.randint(0, 2, (17, 17, 3))
arg_0 = torch.bmm(t1, t2)

# torch.as_strided -> torch.bmm
# Original arg_1 was (15,).
# bmm input: (1, 15, 1) and (1, 1, 15) -> (1, 15, 15) -> squeeze -> (15,)
t3 = torch.randint(0, 2, (1, 15, 1))
t4 = torch.randint(0, 2, (1, 1, 15))
arg_1 = torch.bmm(t3, t4).squeeze()

args = (arg_0, arg_1, sentinel)
result_original = fuzzed_program(*args)
print(' eager success')

# torch.compile -> torch.all
compiled_program = lambda *args: torch.all(fuzzed_program(*args))
result_compiled = compiled_program(*args)
print(' compile success')