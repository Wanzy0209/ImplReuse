import torch
import torch.nn.functional as F

torch._dynamo.config.capture_scalar_outputs = True

# Replacement: torch.manual_seed -> torch.set_rng_state
# We use the current state to ensure the code runs without error, satisfying the replacement.
torch.set_rng_state(torch.get_rng_state())

# Setup arguments to match the function signature
device = 'cuda' if torch.cuda.is_available() else 'cpu'

arg_0 = torch.randint(0, 10, (15, 108, 4), dtype=torch.int16, device=device)
arg_1 = torch.randint(0, 10, (11,), dtype=torch.int64, device=device)
arg_2 = torch.randint(0, 10, (3, 27), dtype=torch.int16, device=device)
arg_3 = torch.randint(0, 10, (1, 27), dtype=torch.int16, device=device)
arg_4 = torch.randint(0, 10, (1, 27), dtype=torch.int16, device=device)
arg_5 = torch.randint(0, 10, (1, 27), dtype=torch.int16, device=device)
arg_6 = torch.randint(0, 10, (1,), dtype=torch.int64, device=device)
sentinel = None

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, sentinel):
    var_node_4 = arg_0

    # Replacement: torch.chunk -> torch.bucketize
    # Original: var_node_3 = torch.chunk(var_node_4, 4, dim=1)[0]
    boundaries = torch.tensor([1, 2, 3], device=var_node_4.device)
    var_node_3 = torch.bucketize(var_node_4, boundaries)

    # Replacement: torch.chunk -> torch.bucketize
    # Original: var_node_2 = torch.chunk(var_node_3, 4, dim=2)[0]
    var_node_2 = torch.bucketize(var_node_3, boundaries)

    # Replacement: torch.squeeze -> torch.nn.ModuleDict
    # Original: var_node_1 = torch.squeeze(var_node_2)
    var_node_1 = torch.nn.ModuleDict()

    # Replacement: torch.full -> torch.fmax
    # Original: var_node_8 = torch.full((13, 27), 3, dtype=torch.int16)
    t1 = torch.zeros((13, 27), dtype=torch.int16, device=var_node_4.device)
    t2 = torch.ones((13, 27), dtype=torch.int16, device=var_node_4.device) * 3
    var_node_8 = torch.fmax(t1, t2)

    var_node_9 = arg_1
    _input_size_var_node_7 = var_node_8.size(0)

    # Replacement: torch.randint -> torch.amax
    # Original: _index_var_node_7 = torch.randint(0, _input_size_var_node_7, (11,), device=var_node_8.device)
    _index_var_node_7 = torch.amax(var_node_8, dim=0)

    # Replacement: torch.index_select -> torch.nn.functional.pdist
    # Original: var_node_7 = torch.index_select(var_node_8, 0, _index_var_node_7)
    var_node_7 = torch.nn.functional.pdist(var_node_8)

    # Replacement: torch.clamp -> torch.clip
    # Original: var_node_6 = torch.clamp(var_node_7, min=-1.0, max=1.0)
    var_node_6 = torch.clip(var_node_7, min=-1.0, max=1.0)

    var_node_12 = arg_2

    # Replacement: torch.clamp -> torch.clip
    # Original: var_node_11 = torch.clamp(var_node_12, min=-1.0, max=1.0)
    var_node_11 = torch.clip(var_node_12, min=-1.0, max=1.0)

    # Replacement: torch.full -> torch.fmax
    # Original: var_node_13 = torch.full((1,), 3, dtype=torch.int64)
    t3 = torch.zeros((1,), dtype=torch.int64, device=var_node_4.device)
    t4 = torch.ones((1,), dtype=torch.int64, device=var_node_4.device) * 3
    var_node_13 = torch.fmax(t3, t4)

    _input_size_var_node_10 = var_node_11.size(0)

    # Replacement: torch.randint -> torch.amax
    # Original: _index_var_node_10 = torch.randint(0, _input_size_var_node_10, (1,), device=var_node_11.device)
    _index_var_node_10 = torch.amax(var_node_11, dim=0)

    # Replacement: torch.index_select -> torch.nn.functional.pdist
    # Original: var_node_10 = torch.index_select(var_node_11, 0, _index_var_node_10)
    var_node_10 = torch.nn.functional.pdist(var_node_11)

    var_node_16 = arg_3
    var_node_17 = arg_4
    var_node_18 = arg_5

    # Replacement: torch.cat -> torch.nn.Linear
    # Original: var_node_15 = torch.cat([var_node_16, var_node_17, var_node_18], dim=0)
    var_node_15 = torch.nn.Linear(27, 27)

    var_node_20 = arg_6

    # Replacement: torch.clamp -> torch.clip
    # Original: var_node_19 = torch.clamp(var_node_20, min=None, max=1.0)
    var_node_19 = torch.clip(var_node_20, min=None, max=1.0)

    # Adaptation for Linear Module to keep code runnable
    _input_size_var_node_14 = var_node_15.weight.size(0)
    
    # Replacement: torch.randint -> torch.amax
    # Original: _index_var_node_14 = torch.randint(0, _input_size_var_node_14, (1,), device=var_node_15.device)
    _index_var_node_14 = torch.amax(var_node_15.weight, dim=0)

    # Replacement: torch.index_select -> torch.nn.functional.pdist
    # Original: var_node_14 = torch.index_select(var_node_15, 0, _index_var_node_14)
    # Using weight tensor to make pdist runnable
    var_node_14 = torch.nn.functional.pdist(var_node_15.weight)

    # Replacement: torch.full -> torch.fmax
    # Original: var_node_22 = torch.full((4, 27), 3, dtype=torch.int16)
    t5 = torch.zeros((4, 27), dtype=torch.int16, device=var_node_4.device)
    t6 = torch.ones((4, 27), dtype=torch.int16, device=var_node_4.device) * 3
    var_node_22 = torch.fmax(t5, t6)

    # Replacement: torch.full -> torch.fmax
    # Original: var_node_24 = torch.full((4,), 3, dtype=torch.int64)
    t7 = torch.zeros((4,), dtype=torch.int64, device=var_node_4.device)
    t8 = torch.ones((4,), dtype=torch.int64, device=var_node_4.device) * 3
    var_node_24 = torch.fmax(t7, t8)

    # Replacement: torch.full -> torch.fmax
    # Original: var_node_25 = torch.full((2,), 3, dtype=torch.int64)
    t9 = torch.zeros((2,), dtype=torch.int64, device=var_node_4.device)
    t10 = torch.ones((2,), dtype=torch.int64, device=var_node_4.device) * 3
    var_node_25 = torch.fmax(t9, t10)

    # --- Test for torch.gather ---
    # Verifying the similar API torch.gather
    src = torch.ones((3, 5), device=var_node_4.device)
    index = torch.tensor([[0, 1, 2, 0, 0], [0, 1, 2, 0, 0], [0, 1, 2, 0, 0]], device=var_node_4.device)
    gathered = torch.gather(src, 1, index)
    assert gathered.shape == (3, 5)

    return var_node_6, var_node_10, var_node_14, var_node_22, var_node_24, var_node_25

fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, sentinel)